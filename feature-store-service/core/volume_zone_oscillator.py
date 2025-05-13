"""
Volume Zone Oscillator Indicator.

This module implements an advanced Volume Zone Oscillator (VZO)
with dynamic thresholds, integration with price action,
and volume trend identification capabilities.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from enum import Enum

from core.base_indicator import BaseIndicator


class AdvancedVZOIndicator(BaseIndicator):
    """
    Advanced Volume Zone Oscillator Indicator.
    
    This indicator implements an enhanced Volume Zone Oscillator (VZO)
    with dynamic thresholds, integration with price action,
    and volume trend identification capabilities.
    """
    
    category = "volume"
    
    def __init__(
        self, 
        period: int = 14,
        signal_period: int = 9,
        volume_factor: float = 1.0,
        threshold_period: int = 50,
        column_close: str = "close",
        column_high: str = "high",
        column_low: str = "low",
        column_volume: str = "volume",
        **kwargs
    ):
        """
        Initialize Advanced VZO indicator.
        
        Args:
            period: Period for VZO calculation
            signal_period: Period for VZO signal line
            volume_factor: Volume adjustment factor
            threshold_period: Period for dynamic threshold calculation
            column_close: Column name for closing prices
            column_high: Column name for high prices
            column_low: Column name for low prices
            column_volume: Column name for volume data
            **kwargs: Additional parameters
        """
        self.period = period
        self.signal_period = signal_period
        self.volume_factor = volume_factor
        self.threshold_period = threshold_period
        self.column_close = column_close
        self.column_high = column_high
        self.column_low = column_low
        self.column_volume = column_volume
        self.name = "advanced_vzo"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced VZO indicator on the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with advanced VZO indicator values
        """
        # Check required columns
        required_columns = [self.column_close, self.column_high, self.column_low, self.column_volume]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Data is missing required columns: {', '.join(missing_columns)}")
            
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate basic VZO
        self._calculate_basic_vzo(result)
        
        # Calculate enhanced VZO components
        self._calculate_enhanced_vzo(result)
        
        return result
    
    def _calculate_basic_vzo(self, data: pd.DataFrame) -> None:
        """
        Calculate the basic Volume Zone Oscillator.
        
        Args:
            data: DataFrame to update with VZO
        """
        # Get close and volume data
        close = data[self.column_close]
        high = data[self.column_high]
        low = data[self.column_low]
        volume = data[self.column_volume]
        
        # Handle missing volume data
        volume = volume.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate price change
        price_change = typical_price.diff()
        
        # Calculate positive and negative volume
        positive_volume = np.where(price_change > 0, volume, 0)
        negative_volume = np.where(price_change < 0, volume, 0)
        
        # Calculate EMAs for positive and negative volume
        positive_volume_ema = pd.Series(positive_volume).ewm(span=self.period, adjust=False).mean()
        negative_volume_ema = pd.Series(negative_volume).ewm(span=self.period, adjust=False).mean()
        
        # Calculate total volume EMA
        total_volume_ema = pd.Series(volume).ewm(span=self.period, adjust=False).mean()
        
        # Calculate VZO
        vzo = ((positive_volume_ema - negative_volume_ema) / 
               (total_volume_ema * self.volume_factor)) * 100
        
        # Calculate VZO signal line
        vzo_signal = vzo.ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculate VZO histogram
        vzo_histogram = vzo - vzo_signal
        
        # Add VZO to dataframe
        data['vzo'] = vzo
        data['vzo_signal'] = vzo_signal
        data['vzo_histogram'] = vzo_histogram
    
    def _calculate_enhanced_vzo(self, data: pd.DataFrame) -> None:
        """
        Calculate enhanced VZO components.
        
        Args:
            data: DataFrame to update with enhanced VZO
        """
        if 'vzo' not in data.columns:
            return
        
        vzo = data['vzo']
        
        # Calculate dynamic thresholds
        threshold_window = min(self.threshold_period, len(data))
        if threshold_window > 0:
            # Calculate upper and lower thresholds based on percentiles
            data['vzo_upper_threshold'] = vzo.rolling(
                window=threshold_window, min_periods=threshold_window // 2
            ).apply(lambda x: np.percentile(x, 80))
            
            data['vzo_lower_threshold'] = vzo.rolling(
                window=threshold_window, min_periods=threshold_window // 2
            ).apply(lambda x: np.percentile(x, 20))
            
            # Fill NaN values with static thresholds
            data['vzo_upper_threshold'].fillna(40, inplace=True)
            data['vzo_lower_threshold'].fillna(-40, inplace=True)
        else:
            # Use static thresholds
            data['vzo_upper_threshold'] = 40
            data['vzo_lower_threshold'] = -40
        
        # Calculate normalized VZO (0-100 scale)
        # This rescales VZO between the dynamic thresholds
        lower_threshold = data['vzo_lower_threshold']
        upper_threshold = data['vzo_upper_threshold']
        
        # Calculate normalized VZO
        range_denominator = upper_threshold - lower_threshold
        data['vzo_normalized'] = np.where(
            range_denominator != 0,
            ((vzo - lower_threshold) / range_denominator) * 100,
            50  # Default value when range is zero
        ).clip(0, 100)
        
        # Calculate VZO trend state
        # 1: Strong bullish, 0: Neutral, -1: Strong bearish
        data['vzo_trend'] = np.where(
            vzo > data['vzo_upper_threshold'],
            1,
            np.where(
                vzo < data['vzo_lower_threshold'],
                -1,
                0
            )
        )
        
        # Calculate VZO divergence with price
        self._calculate_vzo_divergence(data)
        
        # Calculate VZO momentum
        data['vzo_momentum'] = vzo.diff(5)
        
        # Calculate VZO rate of change
        data['vzo_roc'] = vzo.pct_change(5) * 100
    
    def _calculate_vzo_divergence(self, data: pd.DataFrame) -> None:
        """
        Calculate divergence between VZO and price.
        
        Args:
            data: DataFrame to update with VZO divergence
        """
        lookback = min(30, len(data) // 2)
        if lookback < 5:
            return
            
        # Get close prices and VZO
        close = data[self.column_close]
        vzo = data['vzo']
        
        # Initialize divergence columns
        data['vzo_bullish_divergence'] = 0
        data['vzo_bearish_divergence'] = 0
        
        # Need at least lookback points
        if len(data) < lookback:
            return
            
        for i in range(lookback, len(data)):
            # Get windows of data
            window_close = close.iloc[i-lookback:i+1]
            window_vzo = vzo.iloc[i-lookback:i+1]
            
            # Skip if there are NaN values
            if window_close.isna().any() or window_vzo.isna().any():
                continue
                
            # Find recent price lows
            price_lows = self._find_local_extrema(window_close, find_min=True)
            
            # Find recent price highs
            price_highs = self._find_local_extrema(window_close, find_min=False)
            
            # Check for bullish divergence
            # (price making lower lows but VZO making higher lows)
            if len(price_lows) >= 2:
                recent_lows = sorted(price_lows[-2:])
                
                # Get the VZO values at these price lows
                vzo_at_lows = [window_vzo.iloc[idx] for idx in recent_lows]
                
                # Check for bullish divergence
                if (window_close.iloc[recent_lows[1]] < window_close.iloc[recent_lows[0]] and 
                    vzo_at_lows[1] > vzo_at_lows[0] and
                    vzo_at_lows[1] < 0):  # VZO should be in bearish territory
                    
                    # Calculate divergence strength (0-100)
                    price_change = (window_close.iloc[recent_lows[1]] / window_close.iloc[recent_lows[0]] - 1) * 100
                    vzo_change = vzo_at_lows[1] - vzo_at_lows[0]
                    
                    # Stronger when price drops more and VZO rises more
                    div_strength = min(100, max(0, abs(price_change) * abs(vzo_change) / 50))
                    
                    data.iloc[i, data.columns.get_loc('vzo_bullish_divergence')] = div_strength
            
            # Check for bearish divergence
            # (price making higher highs but VZO making lower highs)
            if len(price_highs) >= 2:
                recent_highs = sorted(price_highs[-2:])
                
                # Get the VZO values at these price highs
                vzo_at_highs = [window_vzo.iloc[idx] for idx in recent_highs]
                
                # Check for bearish divergence
                if (window_close.iloc[recent_highs[1]] > window_close.iloc[recent_highs[0]] and 
                    vzo_at_highs[1] < vzo_at_highs[0] and
                    vzo_at_highs[1] > 0):  # VZO should be in bullish territory
                    
                    # Calculate divergence strength (0-100)
                    price_change = (window_close.iloc[recent_highs[1]] / window_close.iloc[recent_highs[0]] - 1) * 100
                    vzo_change = vzo_at_highs[0] - vzo_at_highs[1]
                    
                    # Stronger when price rises more and VZO falls more
                    div_strength = min(100, max(0, abs(price_change) * abs(vzo_change) / 50))
                    
                    data.iloc[i, data.columns.get_loc('vzo_bearish_divergence')] = div_strength
    
    def _find_local_extrema(self, series: pd.Series, find_min: bool = True, window: int = 5) -> List[int]:
        """
        Find local minima or maxima in a time series.
        
        Args:
            series: Time series data
            find_min: If True, find local minima, else find local maxima
            window: Window size for identifying extrema
            
        Returns:
            List of indices of local extrema
        """
        extrema_indices = []
        
        if len(series) < window * 2:
            return extrema_indices
            
        half_window = window // 2
        
        # Iterate through the series
        for i in range(half_window, len(series) - half_window):
            # Get the window around the current point
            window_indices = list(range(i - half_window, i + half_window + 1))
            window_values = [series.iloc[j] for j in window_indices]
            
            # Check if the current point is a local extrema
            if find_min:
                if series.iloc[i] == min(window_values):
                    extrema_indices.append(i)
            else:
                if series.iloc[i] == max(window_values):
                    extrema_indices.append(i)
                    
        return extrema_indices

    def get_vzo_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get VZO analysis for trading decisions.
        
        Args:
            data: DataFrame with calculated VZO data
            
        Returns:
            Dictionary with VZO analysis data
        """
        if data.empty or 'vzo' not in data.columns:
            return {
                'trend': {'state': 'neutral', 'strength': 0},
                'signals': [],
                'divergence': {'bullish': 0, 'bearish': 0}
            }
            
        # Get most recent values
        vzo = data['vzo'].iloc[-1]
        vzo_signal = data['vzo_signal'].iloc[-1]
        vzo_histogram = data['vzo_histogram'].iloc[-1]
        vzo_momentum = data['vzo_momentum'].iloc[-1]
        
        vzo_upper = data['vzo_upper_threshold'].iloc[-1]
        vzo_lower = data['vzo_lower_threshold'].iloc[-1]
        
        bullish_divergence = data['vzo_bullish_divergence'].iloc[-1]
        bearish_divergence = data['vzo_bearish_divergence'].iloc[-1]
        
        # Determine trend state
        if vzo > vzo_upper:
            trend_state = 'strong_bullish'
            trend_strength = min(100, ((vzo - vzo_upper) / 20) * 50 + 50)
        elif vzo > 0:
            trend_state = 'bullish'
            trend_strength = min(100, (vzo / vzo_upper) * 50)
        elif vzo < vzo_lower:
            trend_state = 'strong_bearish'
            trend_strength = min(100, ((vzo_lower - vzo) / 20) * 50 + 50)
        elif vzo < 0:
            trend_state = 'bearish'
            trend_strength = min(100, (vzo / vzo_lower) * 50)
        else:
            trend_state = 'neutral'
            trend_strength = 0
            
        # Generate signals
        signals = []
        
        # Check for crossovers (last 2 points)
        if len(data) >= 2:
            vzo_prev = data['vzo'].iloc[-2]
            signal_prev = data['vzo_signal'].iloc[-2]
            
            # Bullish crossover (VZO crosses above signal)
            if vzo_prev <= signal_prev and vzo > vzo_signal:
                signals.append({
                    'type': 'bullish_crossover',
                    'strength': min(100, abs(vzo - vzo_signal) * 10)
                })
                
            # Bearish crossover (VZO crosses below signal)
            if vzo_prev >= signal_prev and vzo < vzo_signal:
                signals.append({
                    'type': 'bearish_crossover',
                    'strength': min(100, abs(vzo - vzo_signal) * 10)
                })
                
            # Zero line crossover
            if (vzo_prev < 0 and vzo > 0) or (vzo_prev > 0 and vzo < 0):
                signals.append({
                    'type': 'zero_line_crossover',
                    'direction': 'bullish' if vzo > 0 else 'bearish',
                    'strength': min(100, abs(vzo) * 5)
                })
                
        # Check for divergences
        if bullish_divergence > 0:
            signals.append({
                'type': 'bullish_divergence',
                'strength': bullish_divergence
            })
            
        if bearish_divergence > 0:
            signals.append({
                'type': 'bearish_divergence',
                'strength': bearish_divergence
            })
            
        return {
            'trend': {
                'state': trend_state,
                'strength': float(trend_strength),
                'momentum': float(vzo_momentum)
            },
            'signals': signals,
            'divergence': {
                'bullish': float(bullish_divergence),
                'bearish': float(bearish_divergence)
            },
            'levels': {
                'current': float(vzo),
                'signal': float(vzo_signal),
                'upper_threshold': float(vzo_upper),
                'lower_threshold': float(vzo_lower)
            }
        }

    def get_vzo_visualization_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get data for VZO visualization.
        
        Args:
            data: DataFrame with calculated VZO
            
        Returns:
            Dictionary with VZO visualization data
        """
        if data.empty or 'vzo' not in data.columns:
            return {'vzo': [], 'signal': [], 'histogram': [], 'thresholds': {}}
            
        # Get data for visualization
        vzo_data = data['vzo'].fillna(0).tolist()
        signal_data = data['vzo_signal'].fillna(0).tolist()
        histogram_data = data['vzo_histogram'].fillna(0).tolist()
        
        # Get upper and lower thresholds
        upper_thresholds = data['vzo_upper_threshold'].fillna(40).tolist()
        lower_thresholds = data['vzo_lower_threshold'].fillna(-40).tolist()
        
        # Get divergence data
        bullish_div = data['vzo_bullish_divergence'].fillna(0).tolist()
        bearish_div = data['vzo_bearish_divergence'].fillna(0).tolist()
        
        return {
            'vzo': vzo_data,
            'signal': signal_data,
            'histogram': histogram_data,
            'thresholds': {
                'upper': upper_thresholds,
                'lower': lower_thresholds
            },
            'divergence': {
                'bullish': bullish_div,
                'bearish': bearish_div
            }
        }
""""""
