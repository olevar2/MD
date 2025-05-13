"""
Simplified Volatility Indicators for Degraded Mode

This module provides optimized implementations of volatility indicators
designed for high-performance in degraded mode operation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from .base import DegradedModeIndicator, degraded_indicator
from ..base_indicator import BaseIndicator


@degraded_indicator(estimated_speedup=2.5, accuracy_loss=0.1)
class SimplifiedBollingerBands(DegradedModeIndicator):
    """
    Simplified Bollinger Bands implementation optimized for performance.
    
    This implementation uses efficient calculation methods and selective
    computation to reduce processing time while maintaining acceptable accuracy.
    """
    
    def __init__(
        self, 
        name: str = "SimplifiedBollingerBands", 
        window: int = 20, 
        num_std: float = 2.0,
        price_type: str = "close"
    ):
        """
        Initialize simplified Bollinger Bands.
        
        Args:
            name: Indicator name
            window: Window size for moving average
            num_std: Number of standard deviations for bands
            price_type: Price column to use
        """
        super().__init__(name)
        self.window = window
        self.num_std = num_std
        self.price_type = price_type
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate simplified Bollinger Bands using optimized approach.
        
        Args:
            data: Input price data
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with Bollinger Bands values (middle, upper, lower)
        """
        # Override parameters if provided
        window = kwargs.get('window', self.window)
        num_std = kwargs.get('num_std', self.num_std)
        price_type = kwargs.get('price_type', self.price_type)
        
        # Validate inputs
        if price_type not in data.columns:
            raise ValueError(f"Price column '{price_type}' not found in data")
            
        result = data.copy()
        
        # For small datasets, use standard calculation
        if len(data) <= window * 3 or len(data) < 250:
            # Standard Bollinger Bands calculation
            middle_band = data[price_type].rolling(window=window).mean()
            rolling_std = data[price_type].rolling(window=window).std(ddof=0)
            
            upper_band = middle_band + (rolling_std * num_std)
            lower_band = middle_band - (rolling_std * num_std)
            
            result[f'bb_middle_{window}'] = middle_band
            result[f'bb_upper_{window}'] = upper_band
            result[f'bb_lower_{window}'] = lower_band
            result[f'bb_width_{window}'] = (upper_band - lower_band) / middle_band
            
            return result
            
        # For larger datasets, use optimized calculation with sampling
        prices = data[price_type].values
        data_size = len(prices)
        
        # Determine sampling rate based on data size
        sampling_rate = 1
        if data_size > 2000:
            sampling_rate = max(1, min(5, data_size // 2000))
            
        # Pre-allocate arrays
        middle_band = np.zeros_like(prices)
        std_dev = np.zeros_like(prices)
        upper_band = np.zeros_like(prices)
        lower_band = np.zeros_like(prices)
        bandwidth = np.zeros_like(prices)
        
        # Calculate at sampled points
        for i in range(window - 1, data_size, sampling_rate):
            if i >= data_size: break
            
            # Get window data
            window_data = prices[max(0, i - window + 1):i+1]
            
            # Calculate middle band (SMA)
            middle_band[i] = np.mean(window_data)
            
            # Calculate standard deviation
            std_dev[i] = np.std(window_data, ddof=0)
            
            # Calculate upper and lower bands
            upper_band[i] = middle_band[i] + (std_dev[i] * num_std)
            lower_band[i] = middle_band[i] - (std_dev[i] * num_std)
            
            # Calculate bandwidth
            if middle_band[i] != 0:
                bandwidth[i] = (upper_band[i] - lower_band[i]) / middle_band[i]
        
        # If sampling was used, interpolate the missing values
        if sampling_rate > 1:
            self._interpolate_array(middle_band, window - 1, data_size, sampling_rate)
            self._interpolate_array(upper_band, window - 1, data_size, sampling_rate)
            self._interpolate_array(lower_band, window - 1, data_size, sampling_rate)
            self._interpolate_array(bandwidth, window - 1, data_size, sampling_rate)
        
        # Add to result DataFrame
        result[f'bb_middle_{window}'] = middle_band
        result[f'bb_upper_{window}'] = upper_band
        result[f'bb_lower_{window}'] = lower_band
        result[f'bb_width_{window}'] = bandwidth
        
        return result
    
    def _interpolate_array(self, values: np.ndarray, start_idx: int, data_size: int, sampling_rate: int) -> None:
        """
        Interpolate values between sampled points.
        
        Args:
            values: Array to interpolate
            start_idx: Starting index for interpolation
            data_size: Size of the array
            sampling_rate: Sampling rate used
        """
        # Collect valid calculation points
        valid_indices = list(range(start_idx, data_size, sampling_rate))
        if valid_indices and valid_indices[-1] != data_size - 1:
            valid_indices.append(data_size - 1)
            
        # Linear interpolation between valid points
        for i in range(len(valid_indices)-1):
            start_pos = valid_indices[i]
            end_pos = valid_indices[i+1]
            
            if end_pos - start_pos > 1:  # Need interpolation
                start_val = values[start_pos]
                end_val = values[end_pos]
                
                for j in range(start_pos + 1, end_pos):
                    ratio = (j - start_pos) / (end_pos - start_pos)
                    values[j] = start_val + (end_val - start_val) * ratio


@degraded_indicator(estimated_speedup=2.0, accuracy_loss=0.07)
class SimplifiedATR(DegradedModeIndicator):
    """
    Simplified Average True Range implementation optimized for performance.
    
    This implementation uses efficient calculation methods and approximations
    to reduce processing time while maintaining acceptable accuracy.
    """
    
    def __init__(
        self, 
        name: str = "SimplifiedATR", 
        period: int = 14
    ):
        """
        Initialize simplified ATR.
        
        Args:
            name: Indicator name
            period: ATR period
        """
        super().__init__(name)
        self.period = period
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate simplified ATR using optimized approach.
        
        Args:
            data: Input price data (must include high, low, close)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with ATR values
        """
        # Override parameters if provided
        period = kwargs.get('period', self.period)
        
        # Validate inputs - need high, low, close columns
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
                
        result = data.copy()
        
        # For small datasets, use standard calculation
        if len(data) <= period * 3 or len(data) < 250:
            # Calculate True Range
            high = data['high']
            low = data['low']
            close_prev = data['close'].shift(1)
            
            # True Range calculation
            tr1 = high - low
            tr2 = (high - close_prev).abs()
            tr3 = (low - close_prev).abs()
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR using EMA
            atr = true_range.ewm(span=period, adjust=False).mean()
            
            result[f'atr_{period}'] = atr
            return result
            
        # For larger datasets, use optimized calculation with reduced operations
        high_vals = data['high'].values
        low_vals = data['low'].values
        close_vals = data['close'].values
        data_size = len(close_vals)
        
        # Determine sampling rate based on data size
        sampling_rate = 1
        if data_size > 5000:
            sampling_rate = max(1, min(5, data_size // 5000))
            
        # Pre-allocate arrays
        true_range = np.zeros_like(close_vals)
        atr = np.zeros_like(close_vals)
        
        # Calculate true range without creating multiple arrays
        for i in range(1, data_size):
            # Get high-low range
            hl_range = high_vals[i] - low_vals[i]
            
            # Get high-prev_close range
            hc_range = abs(high_vals[i] - close_vals[i-1])
            
            # Get low-prev_close range
            lc_range = abs(low_vals[i] - close_vals[i-1])
            
            # True range is the maximum of these three
            true_range[i] = max(hl_range, hc_range, lc_range)
        
        # Calculate initial ATR as simple average of true range over period
        if period < data_size:
            atr[period] = np.mean(true_range[1:period+1])
            
            # Calculate remaining ATR values using EMA formula
            # Use Wilder's smoothing formula: ATR = ((period-1) * prev_ATR + TR) / period
            for i in range(period + 1, data_size, sampling_rate):
                if i >= data_size: break
                
                atr[i] = ((period - 1) * atr[i-sampling_rate] + true_range[i]) / period
        
        # If sampling was used, interpolate the missing values
        if sampling_rate > 1:
            self._interpolate_array(atr, period, data_size, sampling_rate)
        
        # Add to result DataFrame
        result[f'atr_{period}'] = atr
        
        return result
    
    def _interpolate_array(self, values: np.ndarray, start_idx: int, data_size: int, sampling_rate: int) -> None:
        """
        Interpolate values between sampled points.
        
        Args:
            values: Array to interpolate
            start_idx: Starting index for interpolation
            data_size: Size of the array
            sampling_rate: Sampling rate used
        """
        # Collect valid calculation points
        valid_indices = list(range(start_idx, data_size, sampling_rate))
        if valid_indices and valid_indices[-1] != data_size - 1:
            valid_indices.append(data_size - 1)
            
        # Linear interpolation between valid points
        for i in range(len(valid_indices)-1):
            start_pos = valid_indices[i]
            end_pos = valid_indices[i+1]
            
            if end_pos - start_pos > 1:  # Need interpolation
                start_val = values[start_pos]
                end_val = values[end_pos]
                
                for j in range(start_pos + 1, end_pos):
                    ratio = (j - start_pos) / (end_pos - start_pos)
                    values[j] = start_val + (end_val - start_val) * ratio


class DegradedVolatilityFactory:
    """
    Factory to create appropriate degraded mode volatility indicators.
    """
    
    @staticmethod
    def create(indicator_type: str, **params) -> DegradedModeIndicator:
        """
        Create degraded volatility indicator instance.
        
        Args:
            indicator_type: Type of indicator to create
            **params: Parameters for the indicator
            
        Returns:
            Degraded mode indicator instance
        """
        indicator_type = indicator_type.lower()
        
        if 'bollinger' in indicator_type or 'bb' in indicator_type:
            return SimplifiedBollingerBands(**params)
        elif 'atr' in indicator_type:
            return SimplifiedATR(**params)
        else:
            raise ValueError(f"Unsupported degraded volatility indicator type: {indicator_type}")
