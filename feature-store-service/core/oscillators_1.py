"""
Simplified Oscillator Indicators for Degraded Mode

This module provides optimized implementations of oscillator indicators
designed for high-performance in degraded mode operation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from .base import DegradedModeIndicator, degraded_indicator
from ..base_indicator import BaseIndicator


@degraded_indicator(estimated_speedup=2.5, accuracy_loss=0.08)
class SimplifiedRSI(DegradedModeIndicator):
    """
    Simplified Relative Strength Index implementation optimized for performance.
    
    This implementation uses a more efficient calculation method with fewer
    intermediate arrays and optimized handling of gain/loss calculations.
    """
    
    def __init__(self, name: str = "SimplifiedRSI", period: int = 14, price_type: str = "close"):
        """
        Initialize simplified RSI.
        
        Args:
            name: Indicator name
            period: RSI calculation period
            price_type: Price column to use
        """
        super().__init__(name)
        self.period = period
        self.price_type = price_type
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate simplified RSI using optimized approach.
        
        Args:
            data: Input price data
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with RSI values
        """
        # Override parameters if provided
        period = kwargs.get('period', self.period)
        price_type = kwargs.get('price_type', self.price_type)
        
        # Validate inputs
        if price_type not in data.columns:
            raise ValueError(f"Price column '{price_type}' not found in data")
            
        result = data.copy()
        
        # For small datasets, use standard calculation
        if len(data) <= period * 3 or len(data) < 250:
            # Standard RSI calculation
            delta = data[price_type].diff().dropna()
            gains = delta.copy()
            losses = delta.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = -losses  # Make positive
            
            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            result[f'rsi_{period}'] = rsi
            return result
            
        # For larger datasets, use optimized calculation with sampling
        # Determine sampling rate based on data size
        data_size = len(data)
        sampling_rate = 1
        
        # If data is large, calculate RSI on sampled points then interpolate
        if data_size > 5000:
            sampling_rate = max(1, data_size // 5000)
            
        # Preprocess the data - perform diff calculation once only
        prices = data[price_type].values
        delta = np.zeros_like(prices)
        delta[1:] = prices[1:] - prices[:-1]
        
        # Calculate gains and losses without creating multiple arrays
        # This is more memory efficient
        gains = np.zeros_like(delta)
        losses = np.zeros_like(delta)
        
        # Vectorized operations are faster than Python loops
        mask_gain = delta > 0
        mask_loss = delta < 0
        
        gains[mask_gain] = delta[mask_gain]  
        losses[mask_loss] = -delta[mask_loss]
        
        # Use exponential moving average for efficiency (with RSI alpha adjustment)
        # This is faster than standard simple moving average for large datasets
        alpha = 1.0 / period
        
        # Exponential smoothing 
        avg_gains = np.zeros_like(prices)
        avg_losses = np.zeros_like(prices)
        
        # Initial values are simple averages for first window
        avg_gains[period] = np.mean(gains[1:period+1])
        avg_losses[period] = np.mean(losses[1:period+1])
        
        # Calculate remaining values using EMA formula
        # Only calculate for sampled points to improve performance
        for i in range(period + 1, len(prices), sampling_rate):
            if i >= len(prices): break
            
            avg_gains[i] = (gains[i] * alpha) + (avg_gains[i-sampling_rate] * (1 - alpha))
            avg_losses[i] = (losses[i] * alpha) + (avg_losses[i-sampling_rate] * (1 - alpha))
        
        # If we used sampling, fill in the missing values
        if sampling_rate > 1:
            # Create temporary arrays for interpolation
            valid_indices = list(range(period, len(prices), sampling_rate))
            if valid_indices[-1] != len(prices) - 1:
                valid_indices.append(len(prices) - 1)
                
            # Extract valid values
            valid_avg_gains = avg_gains[valid_indices]
            valid_avg_losses = avg_losses[valid_indices]
            
            # Simple linear interpolation for missing values
            for i in range(len(valid_indices)-1):
                start_idx = valid_indices[i]
                end_idx = valid_indices[i+1]
                
                if end_idx - start_idx > 1:  # Need interpolation
                    start_gain = valid_avg_gains[i]
                    end_gain = valid_avg_gains[i+1]
                    start_loss = valid_avg_losses[i]
                    end_loss = valid_avg_losses[i+1]
                    
                    for j in range(start_idx + 1, end_idx):
                        ratio = (j - start_idx) / (end_idx - start_idx)
                        avg_gains[j] = start_gain + (end_gain - start_gain) * ratio
                        avg_losses[j] = start_loss + (end_loss - start_loss) * ratio
        
        # Calculate RSI values - avoid division by zero
        rs = np.zeros_like(prices)
        rsi = np.zeros_like(prices)
        
        # Only compute where avg_losses > 0 to avoid division by zero
        nonzero_mask = avg_losses > 0
        rs[nonzero_mask] = avg_gains[nonzero_mask] / avg_losses[nonzero_mask]
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        # Handle special case where avg_loss = 0
        rsi[avg_losses == 0] = 100.0
        
        # Set NaN for the initial period
        rsi[:period] = np.nan
        
        # Add to result
        result[f'rsi_{period}'] = rsi
        
        return result


@degraded_indicator(estimated_speedup=3.0, accuracy_loss=0.12)
class SimplifiedMACD(DegradedModeIndicator):
    """
    Simplified Moving Average Convergence Divergence implementation.
    
    This implementation optimizes the calculation process by using 
    exponential smoothing and reducing the number of calculations.
    """
    
    def __init__(
        self, 
        name: str = "SimplifiedMACD", 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9, 
        price_type: str = "close"
    ):
        """
        Initialize simplified MACD.
        
        Args:
            name: Indicator name
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            price_type: Price column to use
        """
        super().__init__(name)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.price_type = price_type
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate simplified MACD using optimized approach.
        
        Args:
            data: Input price data
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with MACD values
        """
        # Override parameters if provided
        fast_period = kwargs.get('fast_period', self.fast_period)
        slow_period = kwargs.get('slow_period', self.slow_period)
        signal_period = kwargs.get('signal_period', self.signal_period)
        price_type = kwargs.get('price_type', self.price_type)
        
        # Validate inputs
        if price_type not in data.columns:
            raise ValueError(f"Price column '{price_type}' not found in data")
            
        result = data.copy()
        
        # For small datasets, use standard calculation
        if len(data) <= slow_period * 3 or len(data) < 250:
            # Standard MACD calculation
            fast_ema = data[price_type].ewm(span=fast_period, adjust=False).mean()
            slow_ema = data[price_type].ewm(span=slow_period, adjust=False).mean()
            macd_line = fast_ema - slow_ema
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            histogram = macd_line - signal_line
            
            result[f'macd_line'] = macd_line
            result[f'macd_signal'] = signal_line
            result[f'macd_histogram'] = histogram
            return result
            
        # For larger datasets, use optimized calculation
        prices = data[price_type].values
        data_size = len(prices)
        
        # Determine if we should use sampling based on data size
        sampling_rate = 1
        if data_size > 5000:
            sampling_rate = max(1, data_size // 5000)
        
        # Calculate fast alpha
        fast_alpha = 2.0 / (fast_period + 1.0)
        # Calculate slow alpha
        slow_alpha = 2.0 / (slow_period + 1.0)
        # Calculate signal alpha
        signal_alpha = 2.0 / (signal_period + 1.0)
        
        # Pre-allocate arrays
        fast_ema = np.zeros_like(prices)
        slow_ema = np.zeros_like(prices)
        macd_line = np.zeros_like(prices)
        signal_line = np.zeros_like(prices)
        histogram = np.zeros_like(prices)
        
        # Initial values - simple average for first window
        fast_ema[fast_period-1] = np.mean(prices[:fast_period])
        slow_ema[slow_period-1] = np.mean(prices[:slow_period])
        
        # Calculate EMA values for both fast and slow periods
        # Calculate for sampled points first for performance
        for i in range(fast_period, data_size, sampling_rate):
            if i >= data_size: break
            
            # Update fast EMA
            if i >= fast_period:
                fast_ema[i] = prices[i] * fast_alpha + fast_ema[i-sampling_rate] * (1 - fast_alpha)
                
            # Update slow EMA
            if i >= slow_period:
                slow_ema[i] = prices[i] * slow_alpha + slow_ema[i-sampling_rate] * (1 - slow_alpha)
                
            # Calculate MACD line where both EMAs are available
            if i >= slow_period:
                macd_line[i] = fast_ema[i] - slow_ema[i]
                
        # Fill in missing values if sampling was used
        if sampling_rate > 1:
            # Interpolate missing values for fast EMA
            self._interpolate_sampled_values(
                fast_ema, fast_period, data_size, sampling_rate
            )
            
            # Interpolate missing values for slow EMA
            self._interpolate_sampled_values(
                slow_ema, slow_period, data_size, sampling_rate
            )
            
            # Calculate MACD line for all points
            for i in range(slow_period, data_size):
                macd_line[i] = fast_ema[i] - slow_ema[i]
        
        # Calculate signal line (EMA of MACD line)
        # Initial value for signal line - simple average of first signal_period MACD values
        signal_start_idx = slow_period + signal_period - 1
        
        if signal_start_idx < data_size:
            # Check if we have enough data points
            signal_line[signal_start_idx] = np.mean(macd_line[slow_period:signal_start_idx+1])
            
            # Calculate remaining signal line values
            for i in range(signal_start_idx + 1, data_size, sampling_rate):
                if i >= data_size: break
                signal_line[i] = macd_line[i] * signal_alpha + signal_line[i-sampling_rate] * (1 - signal_alpha)
                
            # Interpolate signal line if needed
            if sampling_rate > 1:
                self._interpolate_sampled_values(
                    signal_line, signal_start_idx, data_size, sampling_rate
                )
                
            # Calculate histogram
            for i in range(signal_start_idx, data_size):
                histogram[i] = macd_line[i] - signal_line[i]
        
        # Add to result DataFrame
        result['macd_line'] = macd_line
        result['macd_signal'] = signal_line
        result['macd_histogram'] = histogram
        
        return result
    
    def _interpolate_sampled_values(
        self, 
        values: np.ndarray, 
        start_idx: int, 
        data_size: int, 
        sampling_rate: int
    ) -> None:
        """
        Interpolate missing values between sampled points.
        
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


@degraded_indicator(estimated_speedup=2.0, accuracy_loss=0.1)
class SimplifiedStochastic(DegradedModeIndicator):
    """
    Simplified Stochastic Oscillator implementation.
    
    This implementation optimizes calculation by using vectorized operations
    and reducing lookback window calculations.
    """
    
    def __init__(
        self, 
        name: str = "SimplifiedStochastic", 
        k_period: int = 14, 
        d_period: int = 3, 
        slowing: int = 3
    ):
        """
        Initialize simplified Stochastic Oscillator.
        
        Args:
            name: Indicator name
            k_period: K period
            d_period: D period (for %D smoothing)
            slowing: Slowing period
        """
        super().__init__(name)
        self.k_period = k_period
        self.d_period = d_period
        self.slowing = slowing
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate simplified Stochastic Oscillator.
        
        Args:
            data: Input OHLCV data
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with Stochastic Oscillator values
        """
        # Override parameters if provided
        k_period = kwargs.get('k_period', self.k_period)
        d_period = kwargs.get('d_period', self.d_period)
        slowing = kwargs.get('slowing', self.slowing)
        
        # Validate inputs - need high, low, close columns
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
                
        result = data.copy()
        data_size = len(data)
        
        # For small datasets, use standard calculation
        if data_size <= k_period * 3 or data_size < 250:
            # Standard Stochastic calculation
            lowest_low = data['low'].rolling(window=k_period).min()
            highest_high = data['high'].rolling(window=k_period).max()
            
            # %K calculation
            k_fast = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
            
            # Apply slowing if needed
            if slowing > 1:
                k = k_fast.rolling(window=slowing).mean()
            else:
                k = k_fast
                
            # %D calculation
            d = k.rolling(window=d_period).mean()
            
            result['stoch_k'] = k
            result['stoch_d'] = d
            return result
        
        # For larger datasets, use optimized calculation
        # Extract needed columns as numpy arrays for faster calculation
        high_vals = data['high'].values
        low_vals = data['low'].values
        close_vals = data['close'].values
        
        # Determine sampling rate for optimization
        sampling_rate = 1
        if data_size > 5000:
            sampling_rate = max(1, data_size // 5000)
            
        # Pre-allocate arrays
        lowest_low = np.zeros_like(close_vals)
        highest_high = np.zeros_like(close_vals)
        k_fast = np.zeros_like(close_vals)
        k = np.zeros_like(close_vals)
        d = np.zeros_like(close_vals)
        
        # Calculate lowest lows and highest highs at sampled points
        for i in range(k_period - 1, data_size, sampling_rate):
            if i >= data_size: break
            
            # Optimize window calculation
            window_start = max(0, i - k_period + 1)
            
            lowest_low[i] = np.min(low_vals[window_start:i+1])
            highest_high[i] = np.max(high_vals[window_start:i+1])
            
            # Fast %K
            price_range = highest_high[i] - lowest_low[i]
            if price_range > 0:
                k_fast[i] = 100.0 * (close_vals[i] - lowest_low[i]) / price_range
            else:
                # If the range is zero, %K is 100 by convention
                # (price is at the highest level of the period)
                k_fast[i] = 100.0
        
        # If sampling was used, interpolate missing values
        if sampling_rate > 1:
            self._interpolate_array(lowest_low, k_period - 1, data_size, sampling_rate)
            self._interpolate_array(highest_high, k_period - 1, data_size, sampling_rate)
            
            # Recalculate k_fast for interpolated points
            for i in range(k_period - 1, data_size):
                if i % sampling_rate != 0 and i != data_size - 1:
                    price_range = highest_high[i] - lowest_low[i]
                    if price_range > 0:
                        k_fast[i] = 100.0 * (close_vals[i] - lowest_low[i]) / price_range
                    else:
                        k_fast[i] = 100.0
        
        # Apply slowing to %K if needed
        if slowing > 1:
            # Use moving average for slowing
            for i in range(k_period + slowing - 2, data_size, sampling_rate):
                if i >= data_size: break
                k[i] = np.mean(k_fast[i-(slowing-1):i+1])
            
            # Interpolate if sampling was used
            if sampling_rate > 1:
                self._interpolate_array(k, k_period + slowing - 2, data_size, sampling_rate)
        else:
            # No slowing
            k = k_fast.copy()
        
        # Calculate %D (moving average of %K)
        for i in range(k_period + slowing + d_period - 3, data_size, sampling_rate):
            if i >= data_size: break
            d[i] = np.mean(k[i-(d_period-1):i+1])
        
        # Interpolate %D if sampling was used
        if sampling_rate > 1:
            self._interpolate_array(d, k_period + slowing + d_period - 3, data_size, sampling_rate)
        
        # Add to result DataFrame
        result['stoch_k'] = k
        result['stoch_d'] = d
        
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


class DegradedOscillatorFactory:
    """
    Factory to create appropriate degraded mode oscillator indicators.
    """
    
    @staticmethod
    def create(indicator_type: str, **params) -> DegradedModeIndicator:
        """
        Create degraded oscillator instance.
        
        Args:
            indicator_type: Type of indicator to create
            **params: Parameters for the indicator
            
        Returns:
            Degraded mode indicator instance
        """
        indicator_type = indicator_type.lower()
        
        if 'rsi' in indicator_type:
            return SimplifiedRSI(**params)
        elif 'macd' in indicator_type:
            return SimplifiedMACD(**params)
        elif 'stoch' in indicator_type:
            return SimplifiedStochastic(**params)
        else:
            raise ValueError(f"Unsupported degraded oscillator type: {indicator_type}")
