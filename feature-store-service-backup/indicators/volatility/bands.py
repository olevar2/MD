"""
Volatility Bands Module.

This module provides implementations of band-based volatility indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from feature_store_service.indicators.base_indicator import BaseIndicator


class BollingerBands(BaseIndicator):
    """
    Bollinger Bands indicator.
    
    This volatility indicator creates bands around a moving average,
    with the width of the bands varying with volatility.
    """
    
    category = "volatility"
    
    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        column: str = "close",
        **kwargs
    ):
        """
        Initialize Bollinger Bands indicator.
        
        Args:
            window: Lookback period for the moving average
            num_std: Number of standard deviations for the bands
            column: Data column to use for calculations (default: 'close')
            **kwargs: Additional parameters
        """
        self.window = window
        self.num_std = num_std
        self.column = column
        self.name_middle = f"bb_middle_{window}"
        self.name_upper = f"bb_upper_{window}_{num_std}"
        self.name_lower = f"bb_lower_{window}_{num_std}"
        self.name_width = f"bb_width_{window}_{num_std}"
        self.name_pct_b = f"bb_pct_b_{window}_{num_std}"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Bollinger Bands values
        """
        if self.column not in data.columns:
            raise ValueError(f"Data must contain '{self.column}' column")
            
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate middle band (SMA)
        result[self.name_middle] = result[self.column].rolling(window=self.window).mean()
        
        # Calculate standard deviation
        rolling_std = result[self.column].rolling(window=self.window).std()
        
        # Calculate upper and lower bands
        result[self.name_upper] = result[self.name_middle] + (rolling_std * self.num_std)
        result[self.name_lower] = result[self.name_middle] - (rolling_std * self.num_std)
        
        # Calculate bandwidth
        result[self.name_width] = (result[self.name_upper] - result[self.name_lower]) / result[self.name_middle]
        
        # Calculate %B
        result[self.name_pct_b] = (result[self.column] - result[self.name_lower]) / (result[self.name_upper] - result[self.name_lower])
        
        return result


class KeltnerChannels(BaseIndicator):
    """
    Keltner Channels indicator.
    
    This volatility indicator uses ATR to set channel width, creating
    a dynamic envelope around a moving average.
    """
    
    category = "volatility"
    
    def __init__(
        self,
        window: int = 20,
        atr_window: int = 10,
        atr_multiplier: float = 2.0,
        ma_method: str = "ema",
        column: str = "close",
        **kwargs
    ):
        """
        Initialize Keltner Channels indicator.
        
        Args:
            window: Lookback period for the moving average
            atr_window: Lookback period for the ATR calculation
            atr_multiplier: Multiplier for the ATR
            ma_method: Moving average type ('sma' or 'ema')
            column: Data column to use for calculations (default: 'close')
            **kwargs: Additional parameters
        """
        self.window = window
        self.atr_window = atr_window
        self.atr_multiplier = atr_multiplier
        self.ma_method = ma_method
        self.column = column
        self.name_middle = f"kc_middle_{window}_{ma_method}"
        self.name_upper = f"kc_upper_{window}_{atr_window}_{atr_multiplier}"
        self.name_lower = f"kc_lower_{window}_{atr_window}_{atr_multiplier}"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Keltner Channels for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Keltner Channels values
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate middle line
        if self.ma_method == 'sma':
            result[self.name_middle] = result[self.column].rolling(window=self.window).mean()
        elif self.ma_method == 'ema':
            result[self.name_middle] = result[self.column].ewm(span=self.window, adjust=False).mean()
        else:
            raise ValueError(f"Invalid ma_method: {self.ma_method}. Expected 'sma' or 'ema'.")
        
        # Calculate ATR
        high_low = result['high'] - result['low']
        high_close_prev = abs(result['high'] - result['close'].shift(1))
        low_close_prev = abs(result['low'] - result['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_window).mean()
        
        # Calculate upper and lower bands
        result[self.name_upper] = result[self.name_middle] + (atr * self.atr_multiplier)
        result[self.name_lower] = result[self.name_middle] - (atr * self.atr_multiplier)
        
        return result


class DonchianChannels(BaseIndicator):
    """
    Donchian Channels indicator with dynamic period optimization.
    
    This volatility indicator shows the highest high and lowest low over a specified period,
    creating channels that indicate price extremes and potential breakouts.
    """
    
    category = "volatility"
    
    def __init__(
        self,
        window: int = 20,
        optimize_period: bool = False,
        min_period: int = 10,
        max_period: int = 50,
        optimization_metric: str = "volatility_efficiency",
        **kwargs
    ):
        """
        Initialize Donchian Channels indicator.
        
        Args:
            window: Lookback period for the channel calculation
            optimize_period: Whether to dynamically optimize the period
            min_period: Minimum period to consider during optimization
            max_period: Maximum period to consider during optimization
            optimization_metric: Metric to use for optimization ('volatility_efficiency', 
                                'trend_strength', 'dynamic_range')
            **kwargs: Additional parameters
        """
        self.window = window
        self.optimize_period = optimize_period
        self.min_period = min_period
        self.max_period = max_period
        self.optimization_metric = optimization_metric
        self.optimized_window = None
        
        # Define output column names
        self.name_upper = f"donchian_upper_{window}"
        self.name_lower = f"donchian_lower_{window}"
        self.name_middle = f"donchian_middle_{window}"
        self.name_width = f"donchian_width_{window}"
        self.name_optimized_period = "donchian_optimized_period"
        
    def _optimize_period(self, data: pd.DataFrame) -> int:
        """
        Optimize the lookback period based on the selected metric.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Optimized lookback period
        """
        best_period = self.window
        best_score = -np.inf
        
        # Try different periods and evaluate each
        for period in range(self.min_period, self.max_period + 1):
            # Calculate upper and lower bands for this period
            high_max = data['high'].rolling(window=period).max()
            low_min = data['low'].rolling(window=period).min()
            middle = (high_max + low_min) / 2
            width = high_max - low_min
            
            # Calculate score based on the selected metric
            if self.optimization_metric == 'volatility_efficiency':
                # Higher score when channel width captures price movements efficiently
                recent_volatility = data['close'].pct_change().rolling(period).std()
                score = (width / (data['close'] * recent_volatility)).mean()
            
            elif self.optimization_metric == 'trend_strength':
                # Higher score when price stays near the extremes (trending market)
                distance_from_middle = abs(data['close'] - middle)
                score = (distance_from_middle / (width / 2)).mean()
            
            elif self.optimization_metric == 'dynamic_range':
                # Higher score when channel adapts to changing volatility
                volatility_change = width.pct_change().rolling(period).std()
                score = 1 / volatility_change.mean()
            
            else:
                raise ValueError(f"Invalid optimization_metric: {self.optimization_metric}")
                
            # Update best period if we found a better score
            if not np.isnan(score) and score > best_score:
                best_score = score
                best_period = period
                
        return best_period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Donchian Channels for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Donchian Channels values
        """
        required_cols = ['high', 'low']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Optimize period if requested
        if self.optimize_period and len(data) >= self.max_period * 3:  # Need enough data for optimization
            self.optimized_window = self._optimize_period(data)
            window = self.optimized_window
            result[self.name_optimized_period] = self.optimized_window
        else:
            window = self.window
            
        # Calculate channel components
        result[self.name_upper] = result['high'].rolling(window=window).max()
        result[self.name_lower] = result['low'].rolling(window=window).min()
        result[self.name_middle] = (result[self.name_upper] + result[self.name_lower]) / 2
        result[self.name_width] = result[self.name_upper] - result[self.name_lower]
        
        return result