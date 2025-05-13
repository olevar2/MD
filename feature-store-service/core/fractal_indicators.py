"""
Fractal Geometry Indicators Module.

This module provides implementations of fractal-based indicators and
analysis tools, including fractals, Elliott Wave patterns, and other
chaos theory applications to financial markets.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from enum import Enum

from core.base_indicator import BaseIndicator


class FractalIndicator(BaseIndicator):
    """
    Bill Williams Fractal Indicator
    
    Identifies potential reversal points in the market based on
    a 5-bar pattern where the high/low of the middle bar is higher/lower
    than the highs/lows of the surrounding bars.
    
    Bullish Fractal: A low point with two higher lows on each side
    Bearish Fractal: A high point with two lower highs on each side
    """
    
    category = "fractal"
    
    def __init__(
        self, 
        window: int = 2,
        **kwargs
    ):
        """
        Initialize Bill Williams Fractal indicator.
        
        Args:
            window: Number of bars on each side for fractal identification
                   (traditional fractal uses window=2 for 5-bar pattern)
            **kwargs: Additional parameters
        """
        self.window = window
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate fractals for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with fractal values
        """
        required_cols = ['high', 'low']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Initialize fractal columns
        result['fractal_bearish'] = 0
        result['fractal_bullish'] = 0
        
        # We need at least 2*window+1 bars to identify a fractal
        if len(result) < 2 * self.window + 1:
            return result
            
        # Find bearish fractals (high point with lower highs on both sides)
        for i in range(self.window, len(result) - self.window):
            # Check if the current high is higher than all surrounding highs
            is_bearish_fractal = True
            current_high = result['high'].iloc[i]
            
            # Check bars before current
            for j in range(1, self.window + 1):
                if result['high'].iloc[i - j] >= current_high:
                    is_bearish_fractal = False
                    break
                    
            # Check bars after current
            if is_bearish_fractal:
                for j in range(1, self.window + 1):
                    if result['high'].iloc[i + j] >= current_high:
                        is_bearish_fractal = False
                        break
                        
            # Mark bearish fractal if conditions are met
            if is_bearish_fractal:
                result['fractal_bearish'].iloc[i] = 1
                
        # Find bullish fractals (low point with higher lows on both sides)
        for i in range(self.window, len(result) - self.window):
            # Check if the current low is lower than all surrounding lows
            is_bullish_fractal = True
            current_low = result['low'].iloc[i]
            
            # Check bars before current
            for j in range(1, self.window + 1):
                if result['low'].iloc[i - j] <= current_low:
                    is_bullish_fractal = False
                    break
                    
            # Check bars after current
            if is_bullish_fractal:
                for j in range(1, self.window + 1):
                    if result['low'].iloc[i + j] <= current_low:
                        is_bullish_fractal = False
                        break
                        
            # Mark bullish fractal if conditions are met
            if is_bullish_fractal:
                result['fractal_bullish'].iloc[i] = 1
                
        return result
        
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Bill Williams Fractal Indicator',
            'description': 'Identifies potential reversal points using fractal patterns',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'window',
                    'description': 'Number of bars on each side for fractal identification',
                    'type': 'int',
                    'default': 2
                }
            ]
        }


class AlligatorIndicator(BaseIndicator):
    """
    Bill Williams Alligator Indicator
    
    The Alligator Indicator is a combination of three smoothed 
    moving averages (Balance Lines) that help identify trend direction
    and potential entry/exit points.
    
    - Jaw (blue line): 13-period smoothed moving average, shifted 8 periods forward
    - Teeth (red line): 8-period smoothed moving average, shifted 5 periods forward
    - Lips (green line): 5-period smoothed moving average, shifted 3 periods forward
    """
    
    category = "fractal"
    
    def __init__(
        self, 
        jaw_period: int = 13, 
        jaw_shift: int = 8,
        teeth_period: int = 8, 
        teeth_shift: int = 5,
        lips_period: int = 5, 
        lips_shift: int = 3,
        price_source: str = "median",
        **kwargs
    ):
        """
        Initialize Bill Williams Alligator indicator.
        
        Args:
            jaw_period: Period for the Jaw line
            jaw_shift: Forward shift for the Jaw line
            teeth_period: Period for the Teeth line
            teeth_shift: Forward shift for the Teeth line
            lips_period: Period for the Lips line
            lips_shift: Forward shift for the Lips line
            price_source: Source price data ("median", "close", "open", "typical")
            **kwargs: Additional parameters
        """
        self.jaw_period = jaw_period
        self.jaw_shift = jaw_shift
        self.teeth_period = teeth_period
        self.teeth_shift = teeth_shift
        self.lips_period = lips_period
        self.lips_shift = lips_shift
        self.price_source = price_source
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Alligator indicator for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Alligator indicator values
        """
        required_cols = ['high', 'low']
        if self.price_source == "close" and 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column when price_source is 'close'")
        if self.price_source == "open" and 'open' not in data.columns:
            raise ValueError("Data must contain 'open' column when price_source is 'open'")
        if self.price_source == "typical" and ('high' not in data.columns or 
                                             'low' not in data.columns or 
                                             'close' not in data.columns):
            raise ValueError("Data must contain 'high', 'low', and 'close' columns when price_source is 'typical'")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate the source price series
        if self.price_source == "median":
            source = (result['high'] + result['low']) / 2
        elif self.price_source == "close":
            source = result['close']
        elif self.price_source == "open":
            source = result['open']
        elif self.price_source == "typical":
            source = (result['high'] + result['low'] + result['close']) / 3
        else:
            source = (result['high'] + result['low']) / 2  # Default to median price
        
        # Calculate the smoothed moving averages
        # Using SMMA (Smoothed Moving Average) which is a type of EMA with alpha = 1/period
        # For simplicity, we'll use EMA here
        jaw = source.ewm(span=self.jaw_period, adjust=False).mean().shift(self.jaw_shift)
        teeth = source.ewm(span=self.teeth_period, adjust=False).mean().shift(self.teeth_shift)
        lips = source.ewm(span=self.lips_period, adjust=False).mean().shift(self.lips_shift)
        
        # Add the Alligator lines to the result
        result['alligator_jaw'] = jaw
        result['alligator_teeth'] = teeth
        result['alligator_lips'] = lips
        
        return result
        
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Bill Williams Alligator Indicator',
            'description': 'Three smoothed moving averages to identify trend direction',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'jaw_period',
                    'description': 'Period for the Jaw line',
                    'type': 'int',
                    'default': 13
                },
                {
                    'name': 'jaw_shift',
                    'description': 'Forward shift for the Jaw line',
                    'type': 'int',
                    'default': 8
                },
                {
                    'name': 'teeth_period',
                    'description': 'Period for the Teeth line',
                    'type': 'int',
                    'default': 8
                },
                {
                    'name': 'teeth_shift',
                    'description': 'Forward shift for the Teeth line',
                    'type': 'int',
                    'default': 5
                },
                {
                    'name': 'lips_period',
                    'description': 'Period for the Lips line',
                    'type': 'int',
                    'default': 5
                },
                {
                    'name': 'lips_shift',
                    'description': 'Forward shift for the Lips line',
                    'type': 'int',
                    'default': 3
                },
                {
                    'name': 'price_source',
                    'description': 'Source price data',
                    'type': 'string',
                    'default': 'median',
                    'options': ['median', 'close', 'open', 'typical']
                }
            ]
        }


class AwesomeOscillatorFractal(BaseIndicator):
    """
    Bill Williams Awesome Oscillator (Fractal Version)
    
    The Awesome Oscillator is a momentum indicator that shows the 
    difference between a 5-period and 34-period simple moving average
    of the median prices (high+low)/2.
    
    This implementation includes additional fractal-based signals.
    """
    
    category = "fractal"
    
    def __init__(
        self, 
        fast_period: int = 5, 
        slow_period: int = 34,
        **kwargs
    ):
        """
        Initialize Bill Williams Awesome Oscillator indicator.
        
        Args:
            fast_period: Period for the fast moving average
            slow_period: Period for the slow moving average
            **kwargs: Additional parameters
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Awesome Oscillator for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Awesome Oscillator values
        """
        required_cols = ['high', 'low']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate median price (H+L)/2
        median_price = (result['high'] + result['low']) / 2
        
        # Calculate the simple moving averages
        fast_sma = median_price.rolling(window=self.fast_period).mean()
        slow_sma = median_price.rolling(window=self.slow_period).mean()
        
        # Calculate Awesome Oscillator
        ao = fast_sma - slow_sma
        result['ao'] = ao
        
        # Add additional fractal-based signals
        if len(result) >= 3:
            # Calculate AO color (green/red bars)
            result['ao_color'] = np.where(ao > ao.shift(1), 1, -1)
            
            # Calculate saucer signal
            # Saucer: 3 consecutive bars where:
            # - Bearish saucer (sell): red, red, green bars with ascending values
            # - Bullish saucer (buy): green, green, red bars with descending values
            
            # Initialize saucer columns
            result['ao_bullish_saucer'] = 0
            result['ao_bearish_saucer'] = 0
            
            # Calculate saucer signals (start from 3rd bar)
            for i in range(2, len(result)):
                # Get colors and values for the last 3 bars
                color1 = result['ao_color'].iloc[i-2]
                color2 = result['ao_color'].iloc[i-1]
                color3 = result['ao_color'].iloc[i]
                
                ao1 = result['ao'].iloc[i-2]
                ao2 = result['ao'].iloc[i-1]
                ao3 = result['ao'].iloc[i]
                
                # Check for bullish saucer (green, green, red with descending values)
                if color1 == 1 and color2 == 1 and color3 == -1 and ao1 > ao2 and ao2 > ao3 and ao3 > 0:
                    result['ao_bullish_saucer'].iloc[i] = 1
                    
                # Check for bearish saucer (red, red, green with ascending values)
                if color1 == -1 and color2 == -1 and color3 == 1 and ao1 < ao2 and ao2 < ao3 and ao3 < 0:
                    result['ao_bearish_saucer'].iloc[i] = 1
                    
            # Calculate twin peaks signal (two peaks/valleys with second one higher/lower)
            result['ao_bullish_twin_peaks'] = 0
            result['ao_bearish_twin_peaks'] = 0
            
            # Need at least 5 bars for twin peaks
            if len(result) >= 5:
                for i in range(4, len(result)):
                    # Look for local minimums/maximums in AO
                    # This is simplified and would be more robust with actual peak detection
                    
                    # Potential bullish twin peaks (two valleys below zero with second one higher)
                    if (ao.iloc[i-4] > ao.iloc[i-3] < ao.iloc[i-2] > ao.iloc[i-1] < ao.iloc[i] 
                        and ao.iloc[i-3] < 0 and ao.iloc[i-1] < 0 
                        and ao.iloc[i-1] > ao.iloc[i-3]):
                        result['ao_bullish_twin_peaks'].iloc[i] = 1
                        
                    # Potential bearish twin peaks (two peaks above zero with second one lower)
                    if (ao.iloc[i-4] < ao.iloc[i-3] > ao.iloc[i-2] < ao.iloc[i-1] > ao.iloc[i] 
                        and ao.iloc[i-3] > 0 and ao.iloc[i-1] > 0 
                        and ao.iloc[i-1] < ao.iloc[i-3]):
                        result['ao_bearish_twin_peaks'].iloc[i] = 1
                
        return result
        
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Awesome Oscillator (Fractal Version)',
            'description': 'Bill Williams Awesome Oscillator with additional fractal-based signals',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'fast_period',
                    'description': 'Period for the fast moving average',
                    'type': 'int',
                    'default': 5
                },
                {
                    'name': 'slow_period',
                    'description': 'Period for the slow moving average',
                    'type': 'int',
                    'default': 34
                }
            ]
        }


class ElliottWaveAnalyzer(BaseIndicator):
    """
    Elliott Wave Analyzer
    
    Attempts to identify Elliott Wave patterns in price movements.
    Elliott Wave Theory suggests that market prices move in repetitive patterns
    of 5 waves in the direction of the main trend, followed by 3 corrective waves.
    """
    
    category = "fractal"
    
    def __init__(
        self, 
        lookback_period: int = 200,
        min_wave_size: float = 0.01,  # 1% minimum wave size
        zigzag_threshold: float = 0.05,  # 5% for zigzag pivot detection
        **kwargs
    ):
        """
        Initialize Elliott Wave Analyzer.
        
        Args:
            lookback_period: Number of bars to analyze for wave patterns
            min_wave_size: Minimum size of a wave as percentage of price
            zigzag_threshold: Threshold for zigzag pivot detection (percentage)
            **kwargs: Additional parameters
        """
        self.lookback_period = max(40, lookback_period)  # Need enough data for wave analysis
        self.min_wave_size = max(0.001, min_wave_size)
        self.zigzag_threshold = max(0.01, zigzag_threshold)
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Elliott Wave analysis for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Elliott Wave analysis values
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Limit analysis to lookback period
        analysis_data = result.iloc[-self.lookback_period:] if len(result) > self.lookback_period else result
        
        # Initialize Elliott Wave columns
        result['elliott_wave_degree'] = 0  # Wave degree (0 = none, 1-5 = impulse, 6-8 = corrective)
        result['elliott_wave_position'] = 0  # Position in current wave structure
        result['elliott_wave_target'] = 0.0  # Potential price target based on wave measurement
        
        # Find zigzag pivots for wave analysis
        pivots = self._find_zigzag_pivots(analysis_data)
        
        if len(pivots) < 5:
            # Not enough pivots for Elliott Wave analysis
            return result
            
        # Analyze pivots for potential Elliott Wave patterns
        wave_pattern = self._identify_wave_pattern(pivots, analysis_data)
        
        if wave_pattern:
            # Apply the identified wave pattern to the result
            self._apply_wave_pattern(result, wave_pattern)
            
        return result
        
    def _find_zigzag_pivots(self, data: pd.DataFrame) -> List[Tuple[int, float, str]]:
        """
        Find zigzag pivots in price data for wave analysis.
        
        Returns a list of (index, price, type) tuples where type is 'high' or 'low'.
        """
        pivots = []
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        
        # Initialize with the first point as a potential pivot
        current_direction = None
        current_extreme_idx = 0
        current_extreme_price = closes[0]
        current_extreme_type = 'close'
        
        # Identify turning points based on zigzag threshold
        for i in range(1, len(data)):
            # Determine if current close is a significant move from the previous extreme
            price_change = abs(closes[i] - current_extreme_price) / current_extreme_price
            
            if price_change >= self.zigzag_threshold:
                # Significant enough change for a pivot
                if closes[i] > current_extreme_price:
                    # New pivot is higher than previous
                    new_direction = 'up'
                else:
                    # New pivot is lower than previous
                    new_direction = 'down'
                    
                if current_direction is None or new_direction != current_direction:
                    # Direction changed, so the previous extreme was a pivot
                    pivots.append((current_extreme_idx, current_extreme_price, current_extreme_type))
                    
                    # Update direction
                    current_direction = new_direction
                    
                # Update current extreme
                current_extreme_idx = i
                current_extreme_price = closes[i]
                current_extreme_type = 'high' if new_direction == 'up' else 'low'
                
        # Add the last pivot if it's different from the previous one
        if pivots and current_extreme_idx != pivots[-1][0]:
            pivots.append((current_extreme_idx, current_extreme_price, current_extreme_type))
            
        return pivots
        
    def _identify_wave_pattern(
        self, pivots: List[Tuple[int, float, str]], data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Identify potential Elliott Wave patterns in the pivot points.
        
        Returns a dictionary with wave pattern information or None if no pattern is identified.
        """
        # This is a simplified implementation - a full Elliott Wave analysis
        # would be much more complex and include validation of multiple rules
        
        # Need at least 9 pivots for a complete 5-3 wave pattern
        if len(pivots) < 9:
            return None
            
        # Try to identify a 5-wave impulse pattern
        impulse_waves = []
        
        # Start from the most recent pivots and look backwards
        # Typically wave 1-3-5 are in the same direction, and 2-4 are corrective
        
        # Get the most recent 9 pivots (potentially a complete 5-3 wave pattern)
        recent_pivots = pivots[-9:]
        
        # Check if we have a potential 5-wave impulse pattern
        # This is a highly simplified check - real Elliott Wave analysis is much more complex
        
        # Try to detect if we have an impulse wave (5 waves)
        # Typically alternating high-low-high-low-high or low-high-low-high-low
        
        impulse_direction = None
        
        # Determine the direction of the potential impulse wave
        if recent_pivots[0][2] == 'low' and recent_pivots[4][2] == 'high':
            # Potential bullish impulse wave
            impulse_direction = 'bullish'
        elif recent_pivots[0][2] == 'high' and recent_pivots[4][2] == 'low':
            # Potential bearish impulse wave
            impulse_direction = 'bearish'
            
        if impulse_direction:
            # We have a potential impulse wave pattern
            wave_pattern = {
                'type': 'impulse',
                'direction': impulse_direction,
                'pivots': recent_pivots[:5],  # First 5 pivots for the impulse wave
                'corrective_pivots': recent_pivots[5:8],  # Next 3 pivots for correction
                'wave_degrees': [],
                'position': 0,  # Current position in the wave structure
                'targets': []  # Price targets
            }
            
            # Map the wave degrees (1-5 for impulse, A-B-C for corrective)
            if impulse_direction == 'bullish':
                wave_pattern['wave_degrees'] = [1, 2, 3, 4, 5, 'A', 'B', 'C']
            else:
                wave_pattern['wave_degrees'] = [1, 2, 3, 4, 5, 'A', 'B', 'C']
                
            # Determine current position in wave structure
            # For simplicity, we'll just assume we're at the end of the pattern
            wave_pattern['position'] = 8  # After wave C
            
            # Calculate potential targets based on Fibonacci extensions
            # This would use Fibonacci relationships between waves
            
            # Return the identified wave pattern
            return wave_pattern
            
        return None
        
    def _apply_wave_pattern(
        self, result: pd.DataFrame, wave_pattern: Dict[str, Any]
    ) -> None:
        """Apply the identified wave pattern to the result DataFrame."""
        # This is a simplified implementation
        
        # Apply the wave degree and position to the most recent bars
        result['elliott_wave_degree'].iloc[-1] = 8  # End of pattern (wave C)
        result['elliott_wave_position'].iloc[-1] = wave_pattern['position']
        
        # Calculate and apply a potential price target
        # In a real implementation, this would use Fibonacci projections
        # For simplicity, we'll set a basic target
        
        if wave_pattern['direction'] == 'bullish':
            # After a bullish pattern, project a new bullish move
            wave_1_size = wave_pattern['pivots'][1][1] - wave_pattern['pivots'][0][1]
            projected_wave = wave_pattern['corrective_pivots'][2][1] + 1.618 * wave_1_size
            result['elliott_wave_target'].iloc[-1] = projected_wave
        else:
            # After a bearish pattern, project a new bearish move
            wave_1_size = wave_pattern['pivots'][0][1] - wave_pattern['pivots'][1][1]
            projected_wave = wave_pattern['corrective_pivots'][2][1] - 1.618 * wave_1_size
            result['elliott_wave_target'].iloc[-1] = projected_wave
        
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Elliott Wave Analyzer',
            'description': 'Identifies potential Elliott Wave patterns in price movements',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'lookback_period',
                    'description': 'Number of bars to analyze for wave patterns',
                    'type': 'int',
                    'default': 200
                },
                {
                    'name': 'min_wave_size',
                    'description': 'Minimum size of a wave as percentage of price',
                    'type': 'float',
                    'default': 0.01
                },
                {
                    'name': 'zigzag_threshold',
                    'description': 'Threshold for zigzag pivot detection (percentage)',
                    'type': 'float',
                    'default': 0.05
                }
            ]
        }


class HurstExponent(BaseIndicator):
    """
    Hurst Exponent Indicator
    
    The Hurst Exponent quantifies the long-term memory of a time series
    and can be used to determine whether it's trending, mean-reverting, or random.
    
    - H > 0.5: Trending series with persistent behavior
    - H = 0.5: Random walk (no memory)
    - H < 0.5: Mean-reverting series with anti-persistent behavior
    """
    
    category = "fractal"
    
    def __init__(
        self, 
        min_window: int = 10,
        max_window: int = 100,
        step_size: int = 10,
        price_source: str = "close",
        **kwargs
    ):
        """
        Initialize Hurst Exponent indicator.
        
        Args:
            min_window: Minimum window size for R/S analysis
            max_window: Maximum window size for R/S analysis
            step_size: Step size for window increments
            price_source: Source price data to use
            **kwargs: Additional parameters
        """
        self.min_window = max(4, min_window)  # Need at least 4 data points
        self.max_window = max_window
        self.step_size = max(1, step_size)
        self.price_source = price_source
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Hurst Exponent for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Hurst Exponent values
        """
        if self.price_source not in data.columns:
            raise ValueError(f"Price source '{self.price_source}' not found in data")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate the Hurst exponent
        # If there's not enough data for the max window, use the largest possible window
        max_window = min(self.max_window, len(data) // 2)
        
        # Need enough data for at least the minimum window
        if len(data) < self.min_window:
            result['hurst_exponent'] = np.nan
            return result
            
        # Calculate the Hurst exponent for each point
        hurst = np.zeros(len(data))
        
        for i in range(len(data)):
            # For each data point, calculate the Hurst exponent
            # looking back as far as possible up to max_window
            lookback_data = data[self.price_source].iloc[max(0, i - max_window + 1):i + 1]
            
            if len(lookback_data) >= self.min_window:
                hurst[i] = self._calculate_hurst(lookback_data.values)
            else:
                hurst[i] = np.nan
                
        result['hurst_exponent'] = hurst
        
        return result
        
    def _calculate_hurst(self, price_array: np.ndarray) -> float:
        """Calculate the Hurst exponent using the R/S (Rescaled Range) method."""
        # Calculate the log returns
        returns = np.diff(np.log(price_array))
        
        if len(returns) <= self.min_window:
            return np.nan
            
        # Prepare arrays for R/S analysis
        window_sizes = range(self.min_window, min(len(returns), self.max_window), self.step_size)
        rescaled_range = np.zeros(len(window_sizes))
        
        # Calculate R/S values for different window sizes
        for i, window in enumerate(window_sizes):
            rescaled_range[i] = self._calculate_rs(returns, window)
            
        # Calculate the Hurst exponent using log-log regression
        if len(window_sizes) > 1:
            x = np.log(window_sizes)
            y = np.log(rescaled_range)
            
            # Remove any potential NaN values
            mask = ~np.isnan(y)
            if sum(mask) > 1:
                x = x[mask]
                y = y[mask]
                
                # Linear regression to find the Hurst exponent
                slope, _ = np.polyfit(x, y, 1)
                return slope
                
        return np.nan
        
    def _calculate_rs(self, returns: np.ndarray, window: int) -> float:
        """Calculate the Rescaled Range (R/S) for a specific window size."""
        # Calculate how many segments we can divide the returns into
        segments = len(returns) // window
        
        if segments == 0:
            return np.nan
            
        rs_values = np.zeros(segments)
        
        for i in range(segments):
            segment = returns[i * window:(i + 1) * window]
            cumulative = np.cumsum(segment - np.mean(segment))
            r = np.max(cumulative) - np.min(cumulative)  # Range
            s = np.std(segment)  # Standard deviation
            
            # Avoid division by zero
            if s == 0:
                rs_values[i] = 0
            else:
                rs_values[i] = r / s
                
        # Return the average R/S value for this window size
        return np.mean(rs_values)
        
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Hurst Exponent',
            'description': 'Measures long-term memory of a time series',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'min_window',
                    'description': 'Minimum window size for R/S analysis',
                    'type': 'int',
                    'default': 10
                },
                {
                    'name': 'max_window',
                    'description': 'Maximum window size for R/S analysis',
                    'type': 'int',
                    'default': 100
                },
                {
                    'name': 'step_size',
                    'description': 'Step size for window increments',
                    'type': 'int',
                    'default': 10
                },
                {
                    'name': 'price_source',
                    'description': 'Source price data to use',
                    'type': 'string',
                    'default': 'close'
                }
            ]
        }
