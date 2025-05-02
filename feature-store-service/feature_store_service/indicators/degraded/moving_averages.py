"""
Simplified Moving Average Indicators for Degraded Mode

This module provides optimized implementations of moving average indicators
designed for high-performance in degraded mode operation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from .base import DegradedModeIndicator, degraded_indicator
from ..base_indicator import BaseIndicator


@degraded_indicator(estimated_speedup=3.0, accuracy_loss=0.05)
class SimplifiedSMA(DegradedModeIndicator):
    """
    Simplified Simple Moving Average implementation optimized for performance.
    
    This implementation uses downsampling and linear interpolation to reduce
    the computational cost while maintaining reasonable accuracy.
    """
    
    def __init__(self, name: str = "SimplifiedSMA", window: int = 14, price_type: str = "close"):
        """
        Initialize simplified SMA.
        
        Args:
            name: Indicator name
            window: Window size
            price_type: Price column to use
        """
        super().__init__(name)
        self.window = window
        self.price_type = price_type
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate simplified SMA using downsampling for performance.
        
        Args:
            data: Input price data
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with SMA values
        """
        # Override parameters if provided
        window = kwargs.get('window', self.window)
        price_type = kwargs.get('price_type', self.price_type)
        
        # Validate inputs
        if price_type not in data.columns:
            raise ValueError(f"Price column '{price_type}' not found in data")
            
        result = data.copy()
        
        # Determine downsampling factor based on data size and window
        data_size = len(data)
        
        if data_size <= window * 2 or data_size < 1000:
            # For small datasets, use standard calculation
            result[f'sma_{window}'] = data[price_type].rolling(window=window).mean()
            return result
            
        # For larger datasets, use downsampling
        downsample_factor = min(10, max(2, window // 10))
        
        # Downsample the data
        # We'll calculate SMA for fewer points, then interpolate
        index_step = max(1, len(data) // (len(data) // downsample_factor))
        
        # Select indices for calculation
        calc_indices = list(range(0, len(data), index_step))
        # Ensure we include the last points for accuracy
        for i in range(min(window, 20)):
            if len(data) - 1 - i not in calc_indices and len(data) - 1 - i >= 0:
                calc_indices.append(len(data) - 1 - i)
        calc_indices = sorted(set(calc_indices))
        
        # Calculate SMA only for selected indices
        downsampled = data.iloc[calc_indices].copy()
        downsampled[f'sma_{window}'] = downsampled[price_type].rolling(window=window).mean()
        
        # Interpolate the results back to the original index
        # Create a Series with the calculated values
        sparse_sma = pd.Series(
            index=downsampled.index,
            data=downsampled[f'sma_{window}'].values
        )
        
        # Reindex to the original data's index and interpolate
        full_sma = sparse_sma.reindex(data.index).interpolate(method='linear')
        
        # For the initial window where values are NaN, use the standard calculation
        # to avoid interpolation errors
        nan_mask = full_sma.isna()
        if nan_mask.any():
            standard_sma = data[price_type].rolling(window=window).mean()
            full_sma.loc[nan_mask] = standard_sma.loc[nan_mask]
        
        # Add to result
        result[f'sma_{window}'] = full_sma
        
        return result


@degraded_indicator(estimated_speedup=2.5, accuracy_loss=0.08)
class SimplifiedEMA(DegradedModeIndicator):
    """
    Simplified Exponential Moving Average for degraded mode.
    
    Uses a recursive calculation approach with reduced frequency updates
    to improve performance while maintaining reasonable accuracy.
    """
    
    def __init__(self, name: str = "SimplifiedEMA", span: int = 14, price_type: str = "close", alpha: Optional[float] = None):
        """
        Initialize simplified EMA.
        
        Args:
            name: Indicator name
            span: Span for EMA calculation
            price_type: Price column to use
            alpha: Smoothing factor (optional, calculated from span if None)
        """
        super().__init__(name)
        self.span = span
        self.price_type = price_type
        self.alpha = alpha
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate simplified EMA using optimized approach.
        
        Args:
            data: Input price data
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with EMA values
        """
        # Override parameters if provided
        span = kwargs.get('span', self.span)
        price_type = kwargs.get('price_type', self.price_type)
        alpha = kwargs.get('alpha', self.alpha)
        
        # Validate inputs
        if price_type not in data.columns:
            raise ValueError(f"Price column '{price_type}' not found in data")
            
        result = data.copy()
        
        # Calculate alpha from span if not provided
        if alpha is None:
            alpha = 2.0 / (span + 1.0)
        
        data_size = len(data)
        
        if data_size <= span * 2 or data_size < 500:
            # For small datasets, use standard calculation
            result[f'ema_{span}'] = data[price_type].ewm(span=span, adjust=False).mean()
            return result
            
        # For larger datasets, use optimized calculation
        # Calculate initial SMA (typical approach for EMA initialization)
        sma_initial = data[price_type].iloc[:span].mean()
        
        # Determine update frequency based on data size
        update_freq = min(5, max(1, span // 20))
        
        # Pre-allocate results array
        ema_values = np.full(data_size, np.nan)
        
        # Set initial value
        ema_values[span-1] = sma_initial
        
        # Calculate EMA recursively with reduced updates
        last_ema = sma_initial
        
        for i in range(span, data_size):
            if (i - span) % update_freq == 0 or i >= data_size - span:
                # Full calculation for key points
                price = data[price_type].iloc[i]
                current_ema = price * alpha + last_ema * (1 - alpha)
                ema_values[i] = current_ema
                last_ema = current_ema
            else:
                # Skip calculation and use approximation
                # This introduces a small error but significantly improves performance
                ema_values[i] = last_ema
        
        # For frequently updated points, interpolate between them for better smoothness
        if update_freq > 1:
            # Create a Series with the calculated values
            sparse_ema = pd.Series(ema_values, index=data.index)
            
            # Interpolate between explicitly calculated points
            full_ema = sparse_ema.interpolate(method='linear')
            result[f'ema_{span}'] = full_ema
        else:
            result[f'ema_{span}'] = ema_values
            
        return result


class DegradedMovingAverageFactory:
    """
    Factory to create appropriate degraded mode moving average indicators.
    """
    
    @staticmethod
    def create(indicator_type: str, **params) -> DegradedModeIndicator:
        """
        Create degraded moving average instance.
        
        Args:
            indicator_type: Type of indicator to create
            **params: Parameters for the indicator
            
        Returns:
            Degraded mode indicator instance
        """
        indicator_type = indicator_type.lower()
        
        if 'sma' in indicator_type:
            return SimplifiedSMA(**params)
        elif 'ema' in indicator_type:
            return SimplifiedEMA(**params)
        else:
            raise ValueError(f"Unsupported degraded indicator type: {indicator_type}")
