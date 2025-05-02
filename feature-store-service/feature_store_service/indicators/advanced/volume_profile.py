"""
Volume Profile Indicator.

This module implements volume profile analysis with price-volume histogram,
value area calculations, and volume-weighted price distributions.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from enum import Enum
from scipy import stats

from feature_store_service.indicators.base_indicator import BaseIndicator


class VolumeProfileIndicator(BaseIndicator):
    """
    Volume Profile Indicator.
    
    This indicator implements volume profile analysis with price-volume histogram,
    value area calculations, and volume-weighted price distributions.
    """
    
    category = "volume"
    
    def __init__(
        self, 
        num_bins: int = 50,
        value_area_pct: float = 70.0,
        lookback_period: int = 100,
        moving_window: Optional[int] = 20,
        column_high: str = "high",
        column_low: str = "low",
        column_close: str = "close",
        column_volume: str = "volume",
        **kwargs
    ):
        """
        Initialize Volume Profile indicator.
        
        Args:
            num_bins: Number of price bins for the histogram
            value_area_pct: Percentage of volume to include in the value area (default: 70%)
            lookback_period: Period for calculating volume profile
            moving_window: Window size for moving volume profile (None for session-based)
            column_high: Column name for high prices
            column_low: Column name for low prices
            column_close: Column name for closing prices
            column_volume: Column name for volume data
            **kwargs: Additional parameters
        """
        self.num_bins = num_bins
        self.value_area_pct = value_area_pct
        self.lookback_period = lookback_period
        self.moving_window = moving_window
        self.column_high = column_high
        self.column_low = column_low
        self.column_close = column_close
        self.column_volume = column_volume
        self.name = "volume_profile"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume profile on the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume profile analysis
        """
        # Check required columns
        required_columns = [self.column_high, self.column_low, self.column_close, self.column_volume]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Data is missing required columns: {', '.join(missing_columns)}")
            
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate volume profiles
        if self.moving_window is None:
            # Calculate full-period volume profile
            self._calculate_period_volume_profile(result)
        else:
            # Calculate moving volume profile
            self._calculate_moving_volume_profile(result)
        
        return result
    
    def _calculate_period_volume_profile(self, data: pd.DataFrame) -> None:
        """
        Calculate volume profile for the entire period.
        
        Args:
            data: DataFrame to update with volume profile
        """
        if len(data) < 2:
            # Not enough data
            return
        
        # Get price range
        price_max = data[self.column_high].max()
        price_min = data[self.column_low].min()
        price_range = price_max - price_min
        
        if price_range == 0:
            # No price range
            return
        
        # Create price bins
        bin_edges = np.linspace(price_min, price_max, self.num_bins + 1)
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Initialize volume profile
        volume_by_price = np.zeros(self.num_bins)
        
        # Allocate volume to price bins
        for i in range(len(data)):
            high = data[self.column_high].iloc[i]
            low = data[self.column_low].iloc[i]
            volume = data[self.column_volume].iloc[i]
            
            if np.isnan(high) or np.isnan(low) or np.isnan(volume) or volume == 0:
                continue
            
            # Calculate which bins this candle spans
            low_bin = max(0, min(self.num_bins - 1, int((low - price_min) / bin_width)))
            high_bin = max(0, min(self.num_bins - 1, int((high - price_min) / bin_width)))
            
            # Distribute volume across price bins
            if high_bin == low_bin:
                # Candle fits in a single bin
                volume_by_price[low_bin] += volume
            else:
                # Candle spans multiple bins
                # Distribute volume proportionally to the price range in each bin
                for bin_idx in range(low_bin, high_bin + 1):
                    bin_low = max(low, bin_edges[bin_idx])
                    bin_high = min(high, bin_edges[bin_idx + 1])
                    bin_ratio = (bin_high - bin_low) / (high - low)
                    volume_by_price[bin_idx] += volume * bin_ratio
        
        # Calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate Point of Control (POC) - price level with highest volume
        poc_idx = np.argmax(volume_by_price)
        poc_price = bin_centers[poc_idx]
        
        # Calculate Value Area (VA) - price range containing specified % of volume
        total_volume = np.sum(volume_by_price)
        target_volume = total_volume * (self.value_area_pct / 100)
        
        # Start from POC and expand outward
        cumulative_volume = volume_by_price[poc_idx]
        va_high_idx = poc_idx
        va_low_idx = poc_idx
        
        while cumulative_volume < target_volume and (va_low_idx > 0 or va_high_idx < self.num_bins - 1):
            # Determine which direction to expand
            vol_above = volume_by_price[va_high_idx + 1] if va_high_idx < self.num_bins - 1 else 0
            vol_below = volume_by_price[va_low_idx - 1] if va_low_idx > 0 else 0
            
            if vol_above > vol_below and va_high_idx < self.num_bins - 1:
                # Expand upward
                va_high_idx += 1
                cumulative_volume += volume_by_price[va_high_idx]
            elif vol_below > 0 and va_low_idx > 0:
                # Expand downward
                va_low_idx -= 1
                cumulative_volume += volume_by_price[va_low_idx]
            else:
                # Can't expand further
                break
                
        # Calculate Value Area boundaries
        va_high_price = bin_edges[va_high_idx + 1]
        va_low_price = bin_edges[va_low_idx]
        
        # Calculate Volume-Weighted Average Price (VWAP)
        vwap = np.sum(bin_centers * volume_by_price) / total_volume if total_volume > 0 else np.nan
        
        # Store results in DataFrame
        data['volume_profile_poc'] = poc_price
        data['volume_profile_va_high'] = va_high_price
        data['volume_profile_va_low'] = va_low_price
        data['volume_profile_vwap'] = vwap
        
        # Calculate relative price position within Value Area
        current_price = data[self.column_close].iloc[-1]
        if va_high_price > va_low_price:
            va_position = (current_price - va_low_price) / (va_high_price - va_low_price) * 100
            data['volume_profile_va_position'] = va_position.clip(0, 100)
        else:
            data['volume_profile_va_position'] = 50
            
        # Store histogram data for visualization
        self.hist_bins = bin_centers.tolist()
        self.hist_volumes = volume_by_price.tolist()
        self.hist_poc = poc_price
        self.hist_va_high = va_high_price
        self.hist_va_low = va_low_price
    
    def _calculate_moving_volume_profile(self, data: pd.DataFrame) -> None:
        """
        Calculate moving volume profile using a rolling window.
        
        Args:
            data: DataFrame to update with moving volume profile
        """
        # Pre-allocate columns
        data['volume_profile_poc'] = np.nan
        data['volume_profile_va_high'] = np.nan
        data['volume_profile_va_low'] = np.nan
        data['volume_profile_vwap'] = np.nan
        data['volume_profile_va_position'] = np.nan
        
        # Calculate moving volume profile
        window_size = min(self.moving_window, self.lookback_period, len(data))
        
        if window_size < 2:
            # Not enough data
            return
            
        for i in range(window_size, len(data) + 1):
            # Get window data
            window_data = data.iloc[i - window_size:i].copy()
            
            # Calculate volume profile for this window
            price_max = window_data[self.column_high].max()
            price_min = window_data[self.column_low].min()
            price_range = price_max - price_min
            
            if price_range == 0:
                # No price range in this window
                continue
                
            # Create price bins for this window
            bin_edges = np.linspace(price_min, price_max, self.num_bins + 1)
            bin_width = bin_edges[1] - bin_edges[0]
            
            # Initialize volume profile
            volume_by_price = np.zeros(self.num_bins)
            
            # Allocate volume to price bins
            for j in range(len(window_data)):
                high = window_data[self.column_high].iloc[j]
                low = window_data[self.column_low].iloc[j]
                volume = window_data[self.column_volume].iloc[j]
                
                if np.isnan(high) or np.isnan(low) or np.isnan(volume) or volume == 0:
                    continue
                
                # Calculate which bins this candle spans
                low_bin = max(0, min(self.num_bins - 1, int((low - price_min) / bin_width)))
                high_bin = max(0, min(self.num_bins - 1, int((high - price_min) / bin_width)))
                
                # Distribute volume across price bins
                if high_bin == low_bin:
                    # Candle fits in a single bin
                    volume_by_price[low_bin] += volume
                else:
                    # Candle spans multiple bins
                    # Distribute volume proportionally to the price range in each bin
                    for bin_idx in range(low_bin, high_bin + 1):
                        bin_low = max(low, bin_edges[bin_idx])
                        bin_high = min(high, bin_edges[bin_idx + 1])
                        bin_ratio = (bin_high - bin_low) / (high - low)
                        volume_by_price[bin_idx] += volume * bin_ratio
            
            # Calculate bin centers
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Calculate Point of Control (POC) - price level with highest volume
            poc_idx = np.argmax(volume_by_price)
            poc_price = bin_centers[poc_idx]
            
            # Calculate Value Area (VA) - price range containing specified % of volume
            total_volume = np.sum(volume_by_price)
            target_volume = total_volume * (self.value_area_pct / 100)
            
            # Start from POC and expand outward
            cumulative_volume = volume_by_price[poc_idx]
            va_high_idx = poc_idx
            va_low_idx = poc_idx
            
            while cumulative_volume < target_volume and (va_low_idx > 0 or va_high_idx < self.num_bins - 1):
                # Determine which direction to expand
                vol_above = volume_by_price[va_high_idx + 1] if va_high_idx < self.num_bins - 1 else 0
                vol_below = volume_by_price[va_low_idx - 1] if va_low_idx > 0 else 0
                
                if vol_above > vol_below and va_high_idx < self.num_bins - 1:
                    # Expand upward
                    va_high_idx += 1
                    cumulative_volume += volume_by_price[va_high_idx]
                elif vol_below > 0 and va_low_idx > 0:
                    # Expand downward
                    va_low_idx -= 1
                    cumulative_volume += volume_by_price[va_low_idx]
                else:
                    # Can't expand further
                    break
                    
            # Calculate Value Area boundaries
            va_high_price = bin_edges[va_high_idx + 1]
            va_low_price = bin_edges[va_low_idx]
            
            # Calculate Volume-Weighted Average Price (VWAP)
            vwap = np.sum(bin_centers * volume_by_price) / total_volume if total_volume > 0 else np.nan
            
            # Store results for this window's end point
            current_idx = i - 1
            data.iloc[current_idx, data.columns.get_loc('volume_profile_poc')] = poc_price
            data.iloc[current_idx, data.columns.get_loc('volume_profile_va_high')] = va_high_price
            data.iloc[current_idx, data.columns.get_loc('volume_profile_va_low')] = va_low_price
            data.iloc[current_idx, data.columns.get_loc('volume_profile_vwap')] = vwap
            
            # Calculate relative price position within Value Area
            current_price = data[self.column_close].iloc[current_idx]
            if va_high_price > va_low_price:
                va_position = (current_price - va_low_price) / (va_high_price - va_low_price) * 100
                data.iloc[current_idx, data.columns.get_loc('volume_profile_va_position')] = np.clip(va_position, 0, 100)
            else:
                data.iloc[current_idx, data.columns.get_loc('volume_profile_va_position')] = 50
                
            # Store histogram data for the most recent window
            if i == len(data):
                self.hist_bins = bin_centers.tolist()
                self.hist_volumes = volume_by_price.tolist()
                self.hist_poc = poc_price
                self.hist_va_high = va_high_price
                self.hist_va_low = va_low_price

    def get_profile_visualization_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get data for visualizing volume profile.
        
        Args:
            data: DataFrame with calculated volume profile
            
        Returns:
            Dictionary with visualization data
        """
        current_price = data[self.column_close].iloc[-1] if not data.empty else np.nan
        
        visualization_data = {
            'histogram': {
                'bins': getattr(self, 'hist_bins', []),
                'volumes': getattr(self, 'hist_volumes', []),
            },
            'levels': {
                'poc': getattr(self, 'hist_poc', np.nan),
                'va_high': getattr(self, 'hist_va_high', np.nan),
                'va_low': getattr(self, 'hist_va_low', np.nan),
                'current_price': current_price
            },
            'metrics': {
                'va_position': float(data['volume_profile_va_position'].iloc[-1]) if 'volume_profile_va_position' in data.columns else 50.0
            }
        }
        
        return visualization_data
        
    def get_volume_profile_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get trading analysis based on volume profile.
        
        Args:
            data: DataFrame with calculated volume profile
            
        Returns:
            Dictionary with analysis results
        """
        if data.empty or 'volume_profile_poc' not in data.columns:
            return {
                'poc': None,
                'value_area': {'high': None, 'low': None},
                'position': 'unknown',
                'support_resistance': []
            }
        
        # Get latest values
        current_price = data[self.column_close].iloc[-1]
        poc = data['volume_profile_poc'].iloc[-1]
        va_high = data['volume_profile_va_high'].iloc[-1]
        va_low = data['volume_profile_va_low'].iloc[-1]
        
        # Determine price position relative to profile
        if np.isnan(poc) or np.isnan(va_high) or np.isnan(va_low):
            position = 'unknown'
        elif current_price > va_high:
            position = 'above_value_area'
        elif current_price < va_low:
            position = 'below_value_area'
        elif current_price > poc:
            position = 'upper_value_area'
        elif current_price < poc:
            position = 'lower_value_area'
        else:
            position = 'at_poc'
            
        # Identify support and resistance levels
        support_resistance = []
        
        if not np.isnan(poc):
            support_resistance.append({
                'price': float(poc),
                'type': 'poc',
                'strength': 'high'
            })
            
        if not np.isnan(va_high):
            support_resistance.append({
                'price': float(va_high),
                'type': 'resistance',
                'strength': 'medium'
            })
            
        if not np.isnan(va_low):
            support_resistance.append({
                'price': float(va_low),
                'type': 'support',
                'strength': 'medium'
            })
        
        # Return analysis
        return {
            'poc': float(poc) if not np.isnan(poc) else None,
            'value_area': {
                'high': float(va_high) if not np.isnan(va_high) else None,
                'low': float(va_low) if not np.isnan(va_low) else None
            },
            'position': position,
            'support_resistance': support_resistance
        }
"""
