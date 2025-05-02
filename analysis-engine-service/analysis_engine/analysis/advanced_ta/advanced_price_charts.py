"""
Advanced Price Indicators Module

This module provides implementations of advanced price chart types including
Heikin-Ashi, Renko Charts, and Point and Figure Analysis for identifying
trends and filtering market noise.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt

from analysis_engine.analysis.advanced_ta.base import AdvancedAnalysisBase


class HeikinAshi(AdvancedAnalysisBase):
    """
    Heikin-Ashi Analyzer
    
    Implements the Heikin-Ashi ("average bar" in Japanese) technique that uses modified
    candlestick calculations to filter out market noise and better identify trends.
    """
    
    def __init__(
        self,
        name: str = "HeikinAshi",
        trend_threshold: float = 3.0,
        **kwargs
    ):
        """Initialize the Heikin-Ashi analyzer.
        
        Args:
            name: Identifier for this analyzer
            trend_threshold: Threshold for determining strong trends
            **kwargs: Additional parameters
        """
        super().__init__(name=name, **kwargs)
        self.trend_threshold = trend_threshold
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Heikin-Ashi candles.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with Heikin-Ashi values
        """
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        
        # Initialize results with NaN values
        n = len(data)
        ha_open = np.full(n, np.nan)
        ha_close = np.full(n, np.nan)
        ha_high = np.full(n, np.nan)
        ha_low = np.full(n, np.nan)
        
        # Calculate Heikin-Ashi values
        for i in range(n):
            # Calculate HA close first
            ha_close[i] = (data['open'].iloc[i] + data['high'].iloc[i] + 
                          data['low'].iloc[i] + data['close'].iloc[i]) / 4
            
            # Calculate HA open
            if i == 0:
                ha_open[i] = data['open'].iloc[i]
            else:
                ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
                
            # Calculate HA high and low
            ha_high[i] = max(data['high'].iloc[i], ha_open[i], ha_close[i])
            ha_low[i] = min(data['low'].iloc[i], ha_open[i], ha_close[i])
            
        # Convert arrays to Series
        results = {
            'ha_open': pd.Series(ha_open, index=data.index),
            'ha_high': pd.Series(ha_high, index=data.index),
            'ha_low': pd.Series(ha_low, index=data.index),
            'ha_close': pd.Series(ha_close, index=data.index)
        }
        
        # Calculate trend direction
        # Bullish candles (close > open)
        trend = np.zeros(n)
        trend[ha_close > ha_open] = 1  # Bullish
        trend[ha_close < ha_open] = -1  # Bearish
        results['ha_trend'] = pd.Series(trend, index=data.index)
        
        # Calculate body size as percentage of range
        body_size = abs(ha_close - ha_open)
        candle_range = ha_high - ha_low
        strength = body_size / candle_range * 100
        results['ha_strength'] = pd.Series(strength, index=data.index)
        
        # Identify strong trends based on consecutive candles
        consecutive_trend = pd.Series(trend, index=data.index).rolling(window=4).sum()
        strong_trend = np.zeros(n)
        strong_trend[consecutive_trend >= self.trend_threshold] = 2  # Strong bullish
        strong_trend[consecutive_trend <= -self.trend_threshold] = -2  # Strong bearish
        results['ha_strong_trend'] = pd.Series(strong_trend, index=data.index)
        
        self.results = results
        return results


class RenkoCharts(AdvancedAnalysisBase):
    """
    Renko Charts Analyzer
    
    Implements Renko charts that focus solely on price movements of a specified size
    (brick size), filtering out time and minor price fluctuations.
    """
    
    def __init__(
        self, 
        name: str = "RenkoCharts",
        brick_method: str = "atr",
        brick_size: float = 1.0,
        atr_period: int = 14,
        use_color: bool = True,
        **kwargs
    ):
        """Initialize the Renko Charts analyzer.
        
        Args:
            name: Identifier for this analyzer
            brick_method: Method for determining brick size ('fixed' or 'atr')
            brick_size: Size of each brick (fixed) or multiplier (for ATR)
            atr_period: Period for ATR calculation if using ATR method
            use_color: Whether to use colored bricks
            **kwargs: Additional parameters
        """
        super().__init__(name=name, **kwargs)
        self.brick_method = brick_method.lower()
        self.brick_size = brick_size
        self.atr_period = atr_period
        self.use_color = use_color
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Renko Chart data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with Renko Chart values
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        if self.brick_method == 'atr' and (
            'high' not in data.columns or 'low' not in data.columns
        ):
            raise ValueError("Data must contain 'high' and 'low' columns for ATR method")
            
        n = len(data)
        
        # Determine brick size based on method
        if self.brick_method == 'atr':
            # Calculate ATR
            tr = np.zeros(n)
            for i in range(1, n):
                high_low = data['high'].iloc[i] - data['low'].iloc[i]
                high_close_prev = abs(data['high'].iloc[i] - data['close'].iloc[i-1])
                low_close_prev = abs(data['low'].iloc[i] - data['close'].iloc[i-1])
                tr[i] = max(high_low, high_close_prev, low_close_prev)
                
            # Calculate ATR using simple moving average
            atr = pd.Series(tr, index=data.index).rolling(window=self.atr_period).mean()
            brick_size = atr * self.brick_size
        else:  # fixed size method
            brick_size = pd.Series(self.brick_size, index=data.index)
            
        # Initialize Renko data
        renko_price = np.full(n, np.nan)
        renko_direction = np.zeros(n)
        renko_breakout = np.zeros(n)
        
        # Find first valid index with a brick size
        start_idx = 0
        for i in range(n):
            if not np.isnan(brick_size.iloc[i]) and brick_size.iloc[i] > 0:
                start_idx = i
                break
                
        # Initialize the first brick
        if start_idx < n:
            renko_price[start_idx] = data['close'].iloc[start_idx]
            current_price = data['close'].iloc[start_idx]
            last_direction = 0
            
            # Process each subsequent data point
            for i in range(start_idx + 1, n):
                close = data['close'].iloc[i]
                current_brick = brick_size.iloc[i]
                
                if np.isnan(current_brick) or current_brick <= 0:
                    continue
                    
                # Calculate price difference
                price_diff = close - current_price
                
                # Only create new bricks if price moved enough
                if abs(price_diff) >= current_brick:
                    # Determine direction
                    direction = 1 if price_diff > 0 else -1
                    
                    # Number of bricks to create
                    n_bricks = int(abs(price_diff) / current_brick)
                    
                    # Update the price by exact brick multiples
                    current_price += n_bricks * current_brick * direction
                    
                    # Record the new price and direction
                    renko_price[i] = current_price
                    renko_direction[i] = direction
                    
                    # Check for breakout (direction change)
                    if last_direction != 0 and direction != last_direction:
                        renko_breakout[i] = direction
                        
                    last_direction = direction
                else:
                    # No new brick, carry forward the last price
                    renko_price[i] = current_price
        
        # Forward fill prices where no new bricks were created
        renko_price_series = pd.Series(renko_price, index=data.index).ffill()
        
        results = {
            'renko_price': renko_price_series,
            'renko_direction': pd.Series(renko_direction, index=data.index),
            'renko_breakout': pd.Series(renko_breakout, index=data.index),
            'renko_brick_size': brick_size
        }
        
        # Identify key levels
        if len(data) > 30:
            price_series = renko_price_series.dropna()
            
            if not price_series.empty:
                # Count occurrences of each price level
                level_counts = price_series.value_counts()
                
                # Find levels with higher than average occurrence
                mean_count = level_counts.mean()
                key_levels = level_counts[level_counts > mean_count * 1.5].index.tolist()
                
                # Mark key levels in the data
                key_level_markers = np.zeros(n)
                for level in key_levels:
                    key_level_markers[(abs(renko_price_series - level) < 1e-10)] = 1
                    
                results['renko_key_level'] = pd.Series(key_level_markers, index=data.index)
        
        self.results = results
        return results


class PointAndFigure(AdvancedAnalysisBase):
    """
    Point and Figure Analyzer
    
    Implements Point and Figure charts that track price movements of a minimum size
    while ignoring time and smaller price movements, useful for identifying
    support/resistance levels and chart patterns.
    """
    
    def __init__(
        self,
        name: str = "PointAndFigure",
        box_size: Union[float, str] = 'atr',
        reversal_boxes: int = 3,
        atr_period: int = 14,
        source_column: str = 'close',
        atr_multiplier: float = 0.01,
        **kwargs
    ):
        """Initialize the Point and Figure analyzer.
        
        Args:
            name: Identifier for this analyzer
            box_size: Size of each box (fixed value or 'atr')
            reversal_boxes: Number of boxes required for a reversal
            atr_period: Period for ATR calculation if using ATR box sizing
            source_column: Column to use for calculations
            atr_multiplier: Multiplier for ATR if using ATR box sizing
            **kwargs: Additional parameters
        """
        super().__init__(name=name, **kwargs)
        self.box_size_input = box_size
        self.reversal_boxes = reversal_boxes
        self.atr_period = atr_period
        self.source_column = source_column
        self.atr_multiplier = atr_multiplier
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Point and Figure data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with Point and Figure values
        """
        if self.source_column not in data.columns:
            raise ValueError(f"Data must contain '{self.source_column}' column")
            
        # Check if we need ATR for box size
        if self.box_size_input == 'atr':
            if 'high' not in data.columns or 'low' not in data.columns:
                raise ValueError("Data must contain 'high' and 'low' columns for ATR calculation")
        
        n = len(data)
        
        # Determine box size based on method
        if self.box_size_input == 'atr':
            # Calculate ATR
            tr = np.zeros(n)
            for i in range(1, n):
                high_low = data['high'].iloc[i] - data['low'].iloc[i]
                high_close_prev = abs(data['high'].iloc[i] - data['close'].iloc[i-1])
                low_close_prev = abs(data['low'].iloc[i] - data['close'].iloc[i-1])
                tr[i] = max(high_low, high_close_prev, low_close_prev)
                
            # Calculate ATR using simple moving average
            atr = pd.Series(tr, index=data.index).rolling(window=self.atr_period).mean()
            box_size = atr * self.atr_multiplier
        else:  # fixed size
            box_size = pd.Series(float(self.box_size_input), index=data.index)
        
        # Initialize P&F data
        pnf_price = np.full(n, np.nan)
        pnf_direction = np.zeros(n)
        pnf_pattern = np.full(n, None, dtype=object)
        
        # Find first valid index with a box size
        start_idx = 0
        for i in range(n):
            if not np.isnan(box_size.iloc[i]) and box_size.iloc[i] > 0:
                start_idx = i
                break
                
        # Initialize with starting values
        if start_idx < n:
            current_price = data[self.source_column].iloc[start_idx]
            pnf_price[start_idx] = current_price
            
            current_direction = 0  # Start neutral
            box_top = current_price
            box_bottom = current_price
            
            reversal_target_up = 0
            reversal_target_down = 0
            
            # Process each subsequent data point
            for i in range(start_idx + 1, n):
                price = data[self.source_column].iloc[i]
                current_box = box_size.iloc[i]
                
                if np.isnan(current_box) or current_box <= 0:
                    continue
                
                # First move from neutral
                if current_direction == 0:
                    if price >= current_price + current_box:
                        # Move up
                        current_direction = 1
                        box_top = current_price + current_box
                        box_bottom = current_price
                        pnf_direction[i] = 1
                        pnf_price[i] = box_top
                        reversal_target_down = box_top - current_box * self.reversal_boxes
                    elif price <= current_price - current_box:
                        # Move down
                        current_direction = -1
                        box_top = current_price
                        box_bottom = current_price - current_box
                        pnf_direction[i] = -1
                        pnf_price[i] = box_bottom
                        reversal_target_up = box_bottom + current_box * self.reversal_boxes
                    else:
                        # No change
                        pnf_price[i] = current_price
                
                # Continue existing up trend
                elif current_direction == 1:
                    if price >= box_top + current_box:
                        # Add another X box (continue up)
                        box_top += current_box
                        pnf_direction[i] = 1
                        pnf_price[i] = box_top
                        reversal_target_down = box_top - current_box * self.reversal_boxes
                    elif price <= reversal_target_down:
                        # Reversal to O column (down)
                        current_direction = -1
                        box_bottom = box_top - current_box * self.reversal_boxes
                        pnf_direction[i] = -1
                        pnf_price[i] = box_bottom
                        reversal_target_up = box_bottom + current_box * self.reversal_boxes
                        pnf_pattern[i] = "Reversal Down"
                    else:
                        # No change
                        pnf_price[i] = box_top
                
                # Continue existing down trend
                elif current_direction == -1:
                    if price <= box_bottom - current_box:
                        # Add another O box (continue down)
                        box_bottom -= current_box
                        pnf_direction[i] = -1
                        pnf_price[i] = box_bottom
                        reversal_target_up = box_bottom + current_box * self.reversal_boxes
                    elif price >= reversal_target_up:
                        # Reversal to X column (up)
                        current_direction = 1
                        box_top = box_bottom + current_box * self.reversal_boxes
                        pnf_direction[i] = 1
                        pnf_price[i] = box_top
                        reversal_target_down = box_top - current_box * self.reversal_boxes
                        pnf_pattern[i] = "Reversal Up"
                    else:
                        # No change
                        pnf_price[i] = box_bottom
        
        # Forward fill prices where no changes occurred
        pnf_price_series = pd.Series(pnf_price, index=data.index).ffill()
        pnf_direction_series = pd.Series(pnf_direction, index=data.index)
        
        # Create pattern series
        pattern_series = pd.Series([str(x) if x is not None else None for x in pnf_pattern], 
                                   index=data.index)
        
        results = {
            'pnf_price': pnf_price_series,
            'pnf_direction': pnf_direction_series,
            'pnf_box_size': box_size,
            'pnf_pattern': pattern_series
        }
        
        self.results = results
        return results
