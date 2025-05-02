"""
Enhanced Volume Profile Indicator with performance optimizations.

This module implements a high-performance version of the volume profile indicator
using GPU acceleration and memory optimization techniques.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from feature_store_service.indicators.performance_enhanced_indicator import (
    PerformanceEnhancedIndicator, performance_monitored
)
from feature_store_service.optimization.load_balancing import ComputationPriority


class EnhancedVolumeProfile(PerformanceEnhancedIndicator):
    """
    Volume Profile indicator with performance enhancements.
    
    This indicator calculates volume distribution across price levels,
    leveraging GPU acceleration for large datasets.
    """
    
    def __init__(
        self,
        price_bins: int = 100,
        value_area_volume: float = 0.7,
        high_column: str = 'high',
        low_column: str = 'low',
        volume_column: str = 'volume',
        output_prefix: str = 'vol_profile',
        **kwargs
    ):
        """
        Initialize EnhancedVolumeProfile indicator.
        
        Args:
            price_bins: Number of price bins for histogram
            value_area_volume: Percentage of total volume to consider for value area (0-1)
            high_column: Column name for high prices
            low_column: Column name for low prices
            volume_column: Column name for volume
            output_prefix: Prefix for output column names
            **kwargs: Additional keyword arguments for performance options
        """
        super().__init__(**kwargs)
        self.price_bins = price_bins
        self.value_area_volume = value_area_volume
        self.high_column = high_column
        self.low_column = low_column
        self.volume_column = volume_column
        self.output_prefix = output_prefix
        
        # Set appropriate computation priority since this is computation intensive
        self._computation_priority = ComputationPriority.HIGH
    
    def _calculate_impl(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        Calculate Volume Profile with performance optimizations.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with volume profile data
        """
        # Make a copy to avoid modifying the original
        result = data.copy()
        
        # Get input data
        if not all(col in data.columns for col in [self.high_column, self.low_column, self.volume_column]):
            raise ValueError(f"Required columns not found in input data")
            
        high = data[self.high_column].values
        low = data[self.low_column].values
        volume = data[self.volume_column].values
        
        # Use GPU acceleration if available
        if self._gpu_accelerator is not None:
            bin_centers, volumes = self._gpu_accelerator.compute_volume_profile(
                high, low, volume, num_bins=self.price_bins
            )
        else:
            # Standard calculation with numpy
            bin_centers, volumes = self._calculate_volume_profile_numpy(high, low, volume)
        
        # Calculate value area (price range containing self.value_area_volume of total volume)
        value_area = self._calculate_value_area(bin_centers, volumes)
        
        # Add results to DataFrame
        # Store volume profile as a JSON string to avoid creating too many columns
        vol_profile_data = [{
            'price': float(bin_centers[i]),
            'volume': float(volumes[i])
        } for i in range(len(bin_centers))]
        
        result[f'{self.output_prefix}_data'] = str(vol_profile_data)
        result[f'{self.output_prefix}_poc'] = self._find_poc(bin_centers, volumes)
        result[f'{self.output_prefix}_value_area_high'] = value_area['high']
        result[f'{self.output_prefix}_value_area_low'] = value_area['low']
        
        return result
    
    def _calculate_volume_profile_numpy(self, high: np.ndarray, low: np.ndarray, 
                                      volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate volume profile using numpy (CPU implementation).
        
        Args:
            high: Array of high prices
            low: Array of low prices
            volume: Array of volume data
            
        Returns:
            Tuple of (bin_centers, volumes)
        """
        min_price = np.min(low)
        max_price = np.max(high)
        bin_edges = np.linspace(min_price, max_price, self.price_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        volumes = np.zeros(self.price_bins)
        
        for i in range(len(high)):
            # Distribute volume across price range touched by this candle
            h = high[i]
            l = low[i]
            v = volume[i]
            
            # Find which bins this candle spans
            bin_indices = np.where((bin_centers >= l) & (bin_centers <= h))[0]
            if len(bin_indices) > 0:
                # Distribute volume equally across the bins
                volumes[bin_indices] += v / len(bin_indices)
                
        return bin_centers, volumes
    
    def _find_poc(self, bin_centers: np.ndarray, volumes: np.ndarray) -> float:
        """
        Find the Point of Control (price level with highest volume).
        
        Args:
            bin_centers: Array of price levels
            volumes: Array of volume at each price level
            
        Returns:
            Price level with highest volume
        """
        # Find index of maximum volume
        max_idx = np.argmax(volumes)
        return float(bin_centers[max_idx])
    
    def _calculate_value_area(self, bin_centers: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """
        Calculate the Value Area (price range containing self.value_area_volume of total volume).
        
        Args:
            bin_centers: Array of price levels
            volumes: Array of volume at each price level
            
        Returns:
            Dictionary with high and low prices of value area
        """
        total_volume = np.sum(volumes)
        target_volume = total_volume * self.value_area_volume
        
        # Start from POC (Point of Control)
        poc_idx = np.argmax(volumes)
        current_volume = volumes[poc_idx]
        
        # Initialize high and low indices to POC
        high_idx = poc_idx
        low_idx = poc_idx
        
        # Expand from POC until we include enough volume
        while current_volume < target_volume:
            # Check which side has higher volume (up or down from current range)
            high_vol = volumes[high_idx + 1] if high_idx + 1 < len(volumes) else 0
            low_vol = volumes[low_idx - 1] if low_idx - 1 >= 0 else 0
            
            if high_vol >= low_vol and high_idx + 1 < len(volumes):
                # Expand upward
                high_idx += 1
                current_volume += high_vol
            elif low_idx - 1 >= 0:
                # Expand downward
                low_idx -= 1
                current_volume += low_vol
            else:
                # Can't expand further
                break
        
        return {
            'high': float(bin_centers[high_idx]),
            'low': float(bin_centers[low_idx])
        }
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get indicator parameters.
        
        Returns:
            Dictionary with parameters
        """
        return {
            'price_bins': self.price_bins,
            'value_area_volume': self.value_area_volume,
            'high_column': self.high_column,
            'low_column': self.low_column,
            'volume_column': self.volume_column,
            'output_prefix': self.output_prefix,
        }
