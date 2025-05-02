"""
Performance Enhanced Moving Average Indicator.

This module demonstrates how to implement an indicator using the
performance enhanced indicator base class.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from feature_store_service.indicators.performance_enhanced_indicator import (
    PerformanceEnhancedIndicator, performance_monitored
)
from feature_store_service.optimization.load_balancing import ComputationPriority


class EnhancedSMA(PerformanceEnhancedIndicator):
    """
    Simple Moving Average with performance enhancements.
    
    This implementation demonstrates how to leverage GPU acceleration,
    advanced calculation techniques, load balancing, and memory optimization
    for a simple moving average indicator.
    """
    
    def __init__(
        self,
        window: int = 20,
        column: str = 'close',
        output_column: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize EnhancedSMA indicator.
        
        Args:
            window: Window size for moving average
            column: Column name to use for calculation
            output_column: Output column name (defaults to f"sma_{window}")
            **kwargs: Additional keyword arguments for performance options
        """
        super().__init__(**kwargs)
        self.window = window
        self.column = column
        self.output_column = output_column or f"sma_{window}"
        
        # Set appropriate computation priority based on data size and complexity
        if window > 100:
            self._computation_priority = ComputationPriority.HIGH
            
    def _calculate_impl(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        Calculate Simple Moving Average with performance optimizations.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with SMA values
        """
        # Make a copy to avoid modifying the original
        result = data.copy()
        
        # Get input data
        if self.column not in data.columns:
            raise ValueError(f"Column '{self.column}' not found in input data")
            
        input_data = data[self.column].values
        
        # Use GPU acceleration if available
        if self._gpu_accelerator is not None:
            # GPU-accelerated calculation
            sma_values = self._gpu_accelerator.compute_moving_average(input_data, self.window)
        else:
            # Standard calculation with numpy
            sma_values = self._calculate_sma_numpy(input_data)
            
        # Add results to output DataFrame
        result[self.output_column] = sma_values
        
        return result
        
    def _calculate_sma_numpy(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate SMA using numpy (CPU implementation).
        
        Args:
            data: Input array
            
        Returns:
            Array with SMA values
        """
        result = np.full_like(data, np.nan, dtype=np.float64)
        
        # Use numpy's rolling window calculation
        if len(data) >= self.window:
            # Calculate cumulative sum for efficiency
            cumsum = np.cumsum(np.insert(data, 0, 0)) 
            result[self.window-1:] = (cumsum[self.window:] - cumsum[:-self.window]) / self.window
            
        return result
        
    def get_params(self) -> Dict[str, Any]:
        """
        Get indicator parameters.
        
        Returns:
            Dictionary with parameters
        """
        return {
            'window': self.window,
            'column': self.column,
            'output_column': self.output_column
        }
