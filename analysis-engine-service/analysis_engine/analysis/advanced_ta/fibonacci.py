"""
Fibonacci Analysis Module

This module provides comprehensive Fibonacci analysis tools including:
- Fibonacci Retracements
- Fibonacci Extensions
- Fibonacci Arcs
- Fibonacci Fans
- Fibonacci Time Zones
- Fibonacci Circles
- Fibonacci Spirals

Implementations support both standard calculation and incremental updates.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import math

from analysis_engine.analysis.advanced_ta.base import (
    AdvancedAnalysisBase,
    PatternRecognitionBase,
    PatternResult,
    ConfidenceLevel,
    MarketDirection,
    AnalysisTimeframe,
    detect_swings,
    calculate_retracement_levels,
    calculate_projection_levels
)


class FibonacciRetracement(AdvancedAnalysisBase):
    """Fibonacci Retracement calculator"""
    
    def __init__(self, levels: List[float] = None, auto_detect_swings: bool = True, 
                 lookback_period: int = 100, price_column: str = "close"):
        """
        Initialize the Fibonacci Retracement calculator
        
        Args:
            levels: Custom retracement levels (default Fibonacci sequence levels)
            auto_detect_swings: Automatically detect swing highs and lows
            lookback_period: Max periods to look back for swing detection
            price_column: Column name for price data
        """
        parameters = {
            "levels": levels or [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0],
            "auto_detect_swings": auto_detect_swings,
            "lookback_period": lookback_period,
            "price_column": price_column
        }
        super().__init__("Fibonacci Retracement", parameters)
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci retracement levels from swing highs and lows
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Fibonacci retracement levels
        """
        result_df = df.copy()
        
        # Check if we have enough data
        if len(df) < self.parameters["lookback_period"]:
            return result_df
        
        # Use only the lookback period
        analysis_df = df.iloc[-self.parameters["lookback_period"]:]
        price_col = self.parameters["price_column"]
        
        if self.parameters["auto_detect_swings"]:
            # Detect swings
            swing_df = detect_swings(analysis_df, lookback=5, price_col=price_col)
            
            # Find the most recent swing high and low
            swing_highs = swing_df[swing_df["swing_high"]].index
            swing_lows = swing_df[swing_df["swing_low"]].index
            
            if len(swing_highs) > 0 and len(swing_lows) > 0:
                # Find the most recent swing
                latest_high = swing_highs[-1]
                latest_low = swing_lows[-1]
                
                # Determine trend direction based on most recent swing
                if latest_high > latest_low:
                    # Downtrend: high to low
                    start_price = swing_df.loc[latest_high][price_col]
                    end_price = swing_df.loc[latest_low][price_col]
                    trend = MarketDirection.BEARISH
                else:
                    # Uptrend: low to high
                    start_price = swing_df.loc[latest_low][price_col]
                    end_price = swing_df.loc[latest_high][price_col]
                    trend = MarketDirection.BULLISH
                
                # Calculate retracement levels
                levels = self.parameters["levels"]
                retracement_prices = calculate_retracement_levels(start_price, end_price, levels)
                
                # Add levels to the original dataframe
                for level, price in retracement_prices.items():
                    result_df[f"fib_retr_{level}"] = price
                
                # Add trend direction
                result_df["fib_trend"] = trend.name
        
        return result_df
        
    def initialize_incremental(self) -> Dict[str, Any]:
        """
        Initialize state for incremental calculation
        
        Returns:
            State dictionary for incremental updates
        """
        return {
            "price_buffer": [],
            "swings": {"highs": [], "lows": []},
            "current_trend": MarketDirection.NEUTRAL,
            "retracement_levels": {},
            "lookback_period": self.parameters["lookback_period"],
            "price_column": self.parameters["price_column"],
            "levels": self.parameters["levels"],
        }
    
    def update_incremental(self, state: Dict[str, Any], new_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Update Fibonacci retracement levels with new data
        
        Args:
            state: Current calculation state
            new_data: New price data point
            
        Returns:
            Updated state and retracement levels
        """
        # Add new price to buffer
        if self.parameters["price_column"] in new_data:
            price = new_data[self.parameters["price_column"]]
            state["price_buffer"].append(price)
        else:
            return state  # Can't process without the required price data
        
        # Keep buffer within lookback size
        if len(state["price_buffer"]) > state["lookback_period"]:
            state["price_buffer"] = state["price_buffer"][-state["lookback_period"]:]
        
        # Need at least 11 points for swing detection (5 before + current + 5 after)
        if len(state["price_buffer"]) < 11:
            return state
            
        # Check for swing high/low
        # Note: For true swing detection we need future data, so this is approximate
        # For production, use a more sophisticated algorithm or introduce a delay
        mid_point = len(state["price_buffer"]) - 6
        is_swing_high = True
        is_swing_low = True
        
        for i in range(1, 6):
            if state["price_buffer"][mid_point] <= state["price_buffer"][mid_point-i]:
                is_swing_high = False
            if state["price_buffer"][mid_point] >= state["price_buffer"][mid_point-i]:
                is_swing_low = False
                
        # Record swing points
        if is_swing_high:
            state["swings"]["highs"].append((len(state["price_buffer"]) - 6, state["price_buffer"][mid_point]))
            # Keep only recent swings
            if len(state["swings"]["highs"]) > 5:
                state["swings"]["highs"] = state["swings"]["highs"][-5:]
                
        if is_swing_low:
            state["swings"]["lows"].append((len(state["price_buffer"]) - 6, state["price_buffer"][mid_point]))
            # Keep only recent swings
            if len(state["swings"]["lows"]) > 5:
                state["swings"]["lows"] = state["swings"]["lows"][-5:]
        
        # Calculate retracement levels if we have both highs and lows
        if state["swings"]["highs"] and state["swings"]["lows"]:
            latest_high = state["swings"]["highs"][-1]
            latest_low = state["swings"]["lows"][-1]
            
            # Determine trend based on which came last
            if latest_high[0] > latest_low[0]:
                # Downtrend
                start_price = latest_high[1]
                end_price = latest_low[1]
                state["current_trend"] = MarketDirection.BEARISH
            else:
                # Uptrend
                start_price = latest_low[1]
                end_price = latest_high[1]
                state["current_trend"] = MarketDirection.BULLISH
                
            # Calculate retracement levels
            state["retracement_levels"] = calculate_retracement_levels(start_price, end_price, state["levels"])
        
        return state


class FibonacciExtension(AdvancedAnalysisBase):
    """Fibonacci Extension calculator"""
    
    def __init__(self, levels: List[float] = None, auto_detect_swings: bool = True,
                 lookback_period: int = 150, price_column: str = "close",
                 use_three_points: bool = True):
        """
        Initialize the Fibonacci Extension calculator
        
        Args:
            levels: Custom extension levels (default Fibonacci sequence levels)
            auto_detect_swings: Automatically detect swing highs and lows
            lookback_period: Max periods to look back for swing detection
            price_column: Column name for price data
            use_three_points: Use three points for extension calculation (if False, uses two points)
        """
        parameters = {
            "levels": levels or [0.0, 0.382, 0.618, 1.0, 1.382, 1.618, 2.0, 2.618, 3.618, 4.236],
            "auto_detect_swings": auto_detect_swings,
            "lookback_period": lookback_period,
            "price_column": price_column,
            "use_three_points": use_three_points
        }
        super().__init__("Fibonacci Extension", parameters)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci extension levels from swing highs and lows
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Fibonacci extension levels
        """
        result_df = df.copy()
        
        # Check if we have enough data
        if len(df) < self.parameters["lookback_period"]:
            return result_df
        
        # Use only the lookback period
        analysis_df = df.iloc[-self.parameters["lookback_period"]:]
        price_col = self.parameters["price_column"]
        
        if self.parameters["auto_detect_swings"]:
            # Detect swings
            swing_df = detect_swings(analysis_df, lookback=5, price_col=price_col)
            
            # Find recent swing highs and lows
            swing_highs = swing_df[swing_df["swing_high"]].index
            swing_lows = swing_df[swing_df["swing_low"]].index
            
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                # For three-point extension, we need at least 2 swings of each type
                # (For two-point extension, we need only 1 of each, but we're being conservative)
                
                # Sort swings by time
                all_swings = [(idx, "high", swing_df.loc[idx][price_col]) for idx in swing_highs] + \
                             [(idx, "low", swing_df.loc[idx][price_col]) for idx in swing_lows]
                all_swings.sort(key=lambda x: x[0])
                
                # We need at least 3 points for a complete pattern
                if len(all_swings) >= 3:
                    # For uptrend: we want low -> high -> low (points A, B, C)
                    # For downtrend: we want high -> low -> high (points A, B, C)
                    
                    # Find a suitable 3-point pattern
                    pattern_found = False
                    for i in range(len(all_swings) - 2):
                        swing1_type = all_swings[i][1]
                        swing2_type = all_swings[i+1][1]
                        swing3_type = all_swings[i+2][1]
                        
                        # Check if we have valid alternating pattern
                        if (swing1_type == "low" and swing2_type == "high" and swing3_type == "low") or \
                           (swing1_type == "high" and swing2_type == "low" and swing3_type == "high"):
                            # We found a valid pattern
                            pattern_found = True
                            point_a = (all_swings[i][0], all_swings[i][2])
                            point_b = (all_swings[i+1][0], all_swings[i+1][2])
                            point_c = (all_swings[i+2][0], all_swings[i+2][2])
                            trend = MarketDirection.BULLISH if swing1_type == "low" else MarketDirection.BEARISH
                            
                            # For extensions, we'll project from point C in the direction established by A->B
                            if self.parameters["use_three_points"]:
                                # Get price distance from A to B and from B to C
                                ab_range = abs(point_b[1] - point_a[1])
                                bc_range = abs(point_c[1] - point_b[1])
                                
                                # Calculate extension levels based on the last swing (B to C)
                                if trend == MarketDirection.BULLISH:
                                    extension_base = point_c[1]  # Start extensions from point C
                                    levels = self.parameters["levels"]
                                    extension_prices = {level: extension_base + ab_range * level for level in levels}
                                else:  # BEARISH
                                    extension_base = point_c[1]  # Start extensions from point C
                                    levels = self.parameters["levels"]
                                    extension_prices = {level: extension_base - ab_range * level for level in levels}
                            else:
                                # Two-point extension (simpler, just using points B and C)
                                ab_range = abs(point_b[1] - point_a[1])
                                
                                if trend == MarketDirection.BULLISH:
                                    extension_base = point_b[1]  # Start extensions from point B
                                    levels = self.parameters["levels"]
                                    extension_prices = {level: extension_base + ab_range * level for level in levels}
                                else:  # BEARISH
                                    extension_base = point_b[1]  # Start extensions from point B
                                    levels = self.parameters["levels"]
                                    extension_prices = {level: extension_base - ab_range * level for level in levels}
                            
                            # Add levels to the original dataframe
                            for level, price in extension_prices.items():
                                result_df[f"fib_ext_{level}"] = price
                            
                            # Add trend direction and pattern points
                            result_df["fib_ext_trend"] = trend.name
                            result_df["fib_ext_point_a"] = point_a[1]
                            result_df["fib_ext_point_a_idx"] = point_a[0]
                            result_df["fib_ext_point_b"] = point_b[1]
                            result_df["fib_ext_point_b_idx"] = point_b[0]
                            result_df["fib_ext_point_c"] = point_c[1]
                            result_df["fib_ext_point_c_idx"] = point_c[0]
                            
                            # We found a pattern, so we can stop searching
                            break
                
        return result_df
    
    def initialize_incremental(self) -> Dict[str, Any]:
        """
        Initialize state for incremental calculation
        
        Returns:
            State dictionary for incremental updates
        """
        return {
            "price_buffer": [],
            "swings": {"highs": [], "lows": []},
            "current_trend": MarketDirection.NEUTRAL,
            "extension_levels": {},
            "lookback_period": self.parameters["lookback_period"],
            "price_column": self.parameters["price_column"],
            "levels": self.parameters["levels"],
            "pattern_points": {"a": None, "b": None, "c": None}
        }
    
    def update_incremental(self, state: Dict[str, Any], new_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Update Fibonacci extension levels with new data
        
        Args:
            state: Current calculation state
            new_data: New price data point
            
        Returns:
            Updated state and extension levels
        """
        # Add new price to buffer
        if self.parameters["price_column"] in new_data:
            price = new_data[self.parameters["price_column"]]
            state["price_buffer"].append(price)
        else:
            return state  # Can't process without the required price data
        
        # Keep buffer within lookback size
        if len(state["price_buffer"]) > state["lookback_period"]:
            state["price_buffer"] = state["price_buffer"][-state["lookback_period"]:]
            
        # Check if we have enough data points to detect swings
        if len(state["price_buffer"]) > 10:  # Need a minimum number of points
            # Convert buffer to numpy array for faster processing
            prices = np.array(state["price_buffer"])
            
            # Detect new swing highs and lows (using a simple algorithm for incremental update)
            window_size = 5  # Look at 5 points on either side
            for i in range(window_size, len(prices) - window_size):
                window = prices[i-window_size:i+window_size+1]
                current_price = prices[i]
                
                # Check for swing high
                if current_price == np.max(window):
                    # Found a new swing high
                    state["swings"]["highs"].append((len(state["price_buffer"]) - len(prices) + i, current_price))
                
                # Check for swing low
                if current_price == np.min(window):
                    # Found a new swing low
                    state["swings"]["lows"].append((len(state["price_buffer"]) - len(prices) + i, current_price))
                    
            # Keep only the most recent swings to prevent memory growth
            max_swings = 10
            if len(state["swings"]["highs"]) > max_swings:
                state["swings"]["highs"] = state["swings"]["highs"][-max_swings:]
            if len(state["swings"]["lows"]) > max_swings:
                state["swings"]["lows"] = state["swings"]["lows"][-max_swings:]
                
            # Update extension patterns if we have enough swings
            if len(state["swings"]["highs"]) >= 1 and len(state["swings"]["lows"]) >= 1:
                # Sort all swings by time
                all_swings = [(idx, "high", price) for idx, price in state["swings"]["highs"]] + \
                             [(idx, "low", price) for idx, price in state["swings"]["lows"]]
                all_swings.sort(key=lambda x: x[0])
                
                # Look for valid 3-point pattern in the most recent swings
                if len(all_swings) >= 3:
                    # Use the last 3 swing points
                    last_three = all_swings[-3:]
                    
                    # Check if we have valid alternating pattern
                    if (last_three[0][1] != last_three[1][1] and last_three[1][1] != last_three[2][1]):
                        # We have alternating points, determine trend
                        trend = MarketDirection.BULLISH if last_three[0][1] == "low" else MarketDirection.BEARISH
                        
                        # Update pattern points
                        state["pattern_points"]["a"] = (last_three[0][0], last_three[0][2])
                        state["pattern_points"]["b"] = (last_three[1][0], last_three[1][2])
                        state["pattern_points"]["c"] = (last_three[2][0], last_three[2][2])
                        
                        # Calculate extension levels
                        point_a = state["pattern_points"]["a"]
                        point_b = state["pattern_points"]["b"]
                        point_c = state["pattern_points"]["c"]
                        
                        if self.parameters["use_three_points"]:
                            # Three-point extension
                            ab_range = abs(point_b[1] - point_a[1])
                            
                            if trend == MarketDirection.BULLISH:
                                extension_base = point_c[1]
                                state["extension_levels"] = {
                                    level: extension_base + ab_range * level 
                                    for level in state["levels"]
                                }
                            else:  # BEARISH
                                extension_base = point_c[1]
                                state["extension_levels"] = {
                                    level: extension_base - ab_range * level 
                                    for level in state["levels"]
                                }
                        else:
                            # Two-point extension
                            ab_range = abs(point_b[1] - point_a[1])
                            
                            if trend == MarketDirection.BULLISH:
                                extension_base = point_b[1]
                                state["extension_levels"] = {
                                    level: extension_base + ab_range * level 
                                    for level in state["levels"]
                                }
                            else:  # BEARISH
                                extension_base = point_b[1]
                                state["extension_levels"] = {
                                    level: extension_base - ab_range * level 
                                    for level in state["levels"]
                                }
                            
                        state["current_trend"] = trend
                
        return state


class FibonacciArcs(AdvancedAnalysisBase):
    """Fibonacci Arcs calculator
    
    Fibonacci Arcs are half circles that extend out from a trend line at Fibonacci
    levels, creating potential areas of support and resistance based on both price
    and time.
    """
    
    def __init__(self, 
                 levels: List[float] = None,
                 auto_detect_swings: bool = True,
                 lookback_period: int = 100,
                 price_column: str = "close",
                 arc_scaling: float = 1.0,
                 projection_bars: int = 30
                ):
        """
        Initialize the Fibonacci Arcs calculator
        
        Args:
            levels: Custom arc levels (default Fibonacci sequence levels)
            auto_detect_swings: Automatically detect swing highs and lows
            lookback_period: Max periods to look back for swing detection
            price_column: Column name for price data
            arc_scaling: Scaling factor for arc radius calculation
            projection_bars: Number of bars to project arcs into the future
        """
        parameters = {
            "levels": levels or [0.382, 0.5, 0.618, 0.786],
            "auto_detect_swings": auto_detect_swings,
            "lookback_period": lookback_period,
            "price_column": price_column,
            "arc_scaling": arc_scaling,
            "projection_bars": projection_bars
        }
        super().__init__("Fibonacci Arcs", parameters)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci arcs from two significant price points
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Fibonacci arc values
        """
        result_df = df.copy()
        
        # Check if we have enough data
        if len(df) < self.parameters["lookback_period"]:
            return result_df
        
        # Use only the lookback period
        analysis_df = df.iloc[-self.parameters["lookback_period"]:]
        price_col = self.parameters["price_column"]
        
        if self.parameters["auto_detect_swings"]:
            # Detect swings
            swing_df = detect_swings(analysis_df, lookback=5, price_col=price_col)
            
            # Find recent swing highs and lows
            swing_highs = swing_df[swing_df["swing_high"]].index
            swing_lows = swing_df[swing_df["swing_low"]].index
            
            if len(swing_highs) > 0 and len(swing_lows) > 0:
                # Find the most recent swing high and low for trend line
                latest_high = swing_highs[-1]
                latest_low = swing_lows[-1]
                
                # Determine trend direction and select points
                if latest_high > latest_low:  # Most recent is high: low->high trend
                    start_idx = latest_low
                    end_idx = latest_high
                    start_price = swing_df.loc[start_idx][price_col]
                    end_price = swing_df.loc[end_idx][price_col]
                    trend = MarketDirection.BULLISH
                else:  # Most recent is low: high->low trend
                    start_idx = latest_high
                    end_idx = latest_low
                    start_price = swing_df.loc[start_idx][price_col]
                    end_price = swing_df.loc[end_idx][price_col]
                    trend = MarketDirection.BEARISH
                
                # Calculate distance between points (both in bars and price)
                start_loc = df.index.get_loc(start_idx)
                end_loc = df.index.get_loc(end_idx)
                bars_distance = abs(end_loc - start_loc)
                price_distance = abs(end_price - start_price)
                
                # Calculate radius for arcs (using distance between points)
                radius = bars_distance * self.parameters["arc_scaling"]
                
                # Calculate arc values for each level and bar
                for level in self.parameters["levels"]:
                    arc_col = f"fib_arc_{level}"
                    result_df[arc_col] = np.nan
                    
                    # Calculate the level's height based on price range
                    level_height = price_distance * level
                    
                    # For each bar within projection range, calculate arc value
                    projection_range = self.parameters["projection_bars"]
                    last_idx = df.index[-1]
                    last_loc = len(df) - 1
                    
                    # Calculate arc values from end point through projection period
                    for i in range(end_loc, min(last_loc + projection_range + 1, len(df) + projection_bars)):
                        # Distance from end point in bars
                        x_distance = i - end_loc
                        
                        if x_distance <= radius:
                            # Calculate y-coordinate on the arc using circle equation
                            # For a circle: x² + y² = r²
                            # Thus y = sqrt(r² - x²)
                            y_coordinate = math.sqrt(radius**2 - x_distance**2) * level
                            
                            if i < len(df):  # Only set values for existing bars
                                if trend == MarketDirection.BULLISH:
                                    arc_value = end_price - y_coordinate  # Arcs below the end price
                                else:
                                    arc_value = end_price + y_coordinate  # Arcs above the end price
                                    
                                result_df.iloc[i][arc_col] = arc_value
                
                # Add trend direction and arc parameters to result
                result_df["fib_arcs_trend"] = trend.name
                result_df["fib_arcs_start_idx"] = start_idx
                result_df["fib_arcs_end_idx"] = end_idx
                result_df["fib_arcs_start_price"] = start_price
                result_df["fib_arcs_end_price"] = end_price
        
        return result_df
    
    def initialize_incremental(self) -> Dict[str, Any]:
        """Initialize state for incremental calculation"""
        return {
            "price_buffer": [],
            "swings": {"highs": [], "lows": []},
            "current_trend": MarketDirection.NEUTRAL,
            "lookback_period": self.parameters["lookback_period"],
            "price_column": self.parameters["price_column"],
            "levels": self.parameters["levels"],
            "arc_scaling": self.parameters["arc_scaling"],
            "trend_points": {"start": None, "end": None},
            "arc_values": {}
        }
    
    def update_incremental(self, state: Dict[str, Any], new_data: Dict[str, float]) -> Dict[str, Any]:
        """Update Fibonacci arcs with new data"""
        # Add new price to buffer
        if self.parameters["price_column"] in new_data:
            price = new_data[self.parameters["price_column"]]
            state["price_buffer"].append(price)
        else:
            return state
            
        # Keep buffer within lookback size
        if len(state["price_buffer"]) > state["lookback_period"]:
            state["price_buffer"] = state["price_buffer"][-state["lookback_period"]:]
        
        # Detect swings with simplified algorithm
        if len(state["price_buffer"]) > 10:
            prices = np.array(state["price_buffer"])
            window_size = 5
            
            # Check only the last few points for new swings
            check_range = min(10, len(prices) - window_size)
            for i in range(len(prices) - check_range, len(prices) - window_size):
                window = prices[i-window_size:i+window_size+1]
                current_price = prices[i]
                
                # Check for swing high
                if current_price == np.max(window):
                    state["swings"]["highs"].append((len(state["price_buffer"]) - len(prices) + i, current_price))
                
                # Check for swing low
                if current_price == np.min(window):
                    state["swings"]["lows"].append((len(state["price_buffer"]) - len(prices) + i, current_price))
            
            # Limit stored swings
            max_swings = 10
            if len(state["swings"]["highs"]) > max_swings:
                state["swings"]["highs"] = state["swings"]["highs"][-max_swings:]
            if len(state["swings"]["lows"]) > max_swings:
                state["swings"]["lows"] = state["swings"]["lows"][-max_swings:]
            
            # Update trend points if we have both highs and lows
            if state["swings"]["highs"] and state["swings"]["lows"]:
                latest_high = state["swings"]["highs"][-1]
                latest_low = state["swings"]["lows"][-1]
                
                # Determine trend
                if latest_high[0] > latest_low[0]:  # Most recent is high
                    state["trend_points"]["start"] = latest_low
                    state["trend_points"]["end"] = latest_high
                    state["current_trend"] = MarketDirection.BULLISH
                else:  # Most recent is low
                    state["trend_points"]["start"] = latest_high
                    state["trend_points"]["end"] = latest_low
                    state["current_trend"] = MarketDirection.BEARISH
                
                # Calculate arcs for each level
                start_point = state["trend_points"]["start"]
                end_point = state["trend_points"]["end"]
                
                if start_point and end_point:
                    bars_distance = abs(end_point[0] - start_point[0])
                    price_distance = abs(end_point[1] - start_point[1])
                    
                    # Calculate radius for arcs
                    radius = bars_distance * state["arc_scaling"]
                    
                    # Update arc values
                    state["arc_values"] = {
                        level: {
                            "radius": radius,
                            "level_height": price_distance * level,
                            "end_point": end_point,
                            "price_distance": price_distance
                        }
                        for level in state["levels"]
                    }
        
        return state


class FibonacciFans(AdvancedAnalysisBase):
    """Fibonacci Fans calculator
    
    Fibonacci Fans consist of diagonal lines drawn from a price pivot point that pass through 
    invisible horizontal Fibonacci retracement levels, creating potential support and resistance lines.
    """
    
    def __init__(self, 
                 levels: List[float] = None,
                 auto_detect_swings: bool = True,
                 lookback_period: int = 100,
                 price_column: str = "close",
                 projection_bars: int = 30
                ):
        """
        Initialize the Fibonacci Fans calculator
        
        Args:
            levels: Custom fan levels (default Fibonacci sequence levels)
            auto_detect_swings: Automatically detect swing highs and lows
            lookback_period: Max periods to look back for swing detection
            price_column: Column name for price data
            projection_bars: Number of bars to project fans into the future
        """
        parameters = {
            "levels": levels or [0.382, 0.5, 0.618, 0.786],
            "auto_detect_swings": auto_detect_swings,
            "lookback_period": lookback_period,
            "price_column": price_column,
            "projection_bars": projection_bars
        }
        super().__init__("Fibonacci Fans", parameters)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci fans from two significant price points
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Fibonacci fan values
        """
        result_df = df.copy()
        
        # Check if we have enough data
        if len(df) < self.parameters["lookback_period"]:
            return result_df
        
        # Use only the lookback period
        analysis_df = df.iloc[-self.parameters["lookback_period"]:]
        price_col = self.parameters["price_column"]
        
        if self.parameters["auto_detect_swings"]:
            # Detect swings
            swing_df = detect_swings(analysis_df, lookback=5, price_col=price_col)
            
            # Find recent swing highs and lows
            swing_highs = swing_df[swing_df["swing_high"]].index
            swing_lows = swing_df[swing_df["swing_low"]].index
            
            if len(swing_highs) > 0 and len(swing_lows) > 0:
                # Find the most recent significant swing high and low
                latest_high = swing_highs[-1]
                latest_low = swing_lows[-1]
                
                # Determine pivot point and trend direction
                if latest_high > latest_low:  # Most recent is high: uptrend then reversal
                    pivot_idx = latest_high
                    secondary_idx = latest_low
                    pivot_price = swing_df.loc[pivot_idx][price_col]
                    secondary_price = swing_df.loc[secondary_idx][price_col]
                    trend = MarketDirection.BEARISH  # Fans drawn downward from high
                else:  # Most recent is low: downtrend then reversal
                    pivot_idx = latest_low
                    secondary_idx = latest_high
                    pivot_price = swing_df.loc[pivot_idx][price_col]
                    secondary_price = swing_df.loc[secondary_idx][price_col]
                    trend = MarketDirection.BULLISH  # Fans drawn upward from low
                
                # Get locations in the dataframe
                pivot_loc = df.index.get_loc(pivot_idx)
                secondary_loc = df.index.get_loc(secondary_idx)
                
                # Calculate price range and bar distance
                price_range = abs(pivot_price - secondary_price)
                bar_distance = abs(pivot_loc - secondary_loc)
                
                # Calculate fan values for each level and bar
                for level in self.parameters["levels"]:
                    fan_col = f"fib_fan_{level}"
                    result_df[fan_col] = np.nan
                    
                    # For each bar after the pivot point, calculate the fan line value
                    projection_range = self.parameters["projection_bars"]
                    last_idx = len(df) - 1
                    
                    # Start from pivot and go forward
                    for i in range(pivot_loc, min(last_idx + projection_range + 1, len(df) + projection_bars)):
                        # Distance from pivot in bars
                        x_distance = i - pivot_loc
                        
                        # Calculate slope of the fan line based on level
                        if trend == MarketDirection.BULLISH:
                            # Fan lines go up from pivot (low)
                            # Level determines slope: higher level means steeper slope
                            fan_value = pivot_price + (price_range * level * x_distance / bar_distance)
                        else:  # BEARISH
                            # Fan lines go down from pivot (high)
                            fan_value = pivot_price - (price_range * level * x_distance / bar_distance)
                        
                        # Set value if within existing data
                        if i < len(df):
                            result_df.iloc[i][fan_col] = fan_value
                
                # Add trend direction and fan parameters to result
                result_df["fib_fans_trend"] = trend.name
                result_df["fib_fans_pivot_idx"] = pivot_idx
                result_df["fib_fans_pivot_price"] = pivot_price
                result_df["fib_fans_secondary_idx"] = secondary_idx
                result_df["fib_fans_secondary_price"] = secondary_price
        
        return result_df
    
    def initialize_incremental(self) -> Dict[str, Any]:
        """Initialize state for incremental calculation"""
        return {
            "price_buffer": [],
            "swings": {"highs": [], "lows": []},
            "current_trend": MarketDirection.NEUTRAL,
            "lookback_period": self.parameters["lookback_period"],
            "price_column": self.parameters["price_column"],
            "levels": self.parameters["levels"],
            "pivot_point": None,
            "secondary_point": None,
            "fan_values": {}
        }
    
    def update_incremental(self, state: Dict[str, Any], new_data: Dict[str, float]) -> Dict[str, Any]:
        """Update Fibonacci fans with new data"""
        # Add new price to buffer
        if self.parameters["price_column"] in new_data:
            price = new_data[self.parameters["price_column"]]
            state["price_buffer"].append(price)
        else:
            return state
            
        # Keep buffer within lookback size
        if len(state["price_buffer"]) > state["lookback_period"]:
            state["price_buffer"] = state["price_buffer"][-state["lookback_period"]:]
        
        # Detect swings with simplified algorithm
        if len(state["price_buffer"]) > 10:
            prices = np.array(state["price_buffer"])
            window_size = 5
            
            # Check only the last few points for new swings
            check_range = min(10, len(prices) - window_size)
            for i in range(len(prices) - check_range, len(prices) - window_size):
                window = prices[i-window_size:i+window_size+1]
                current_price = prices[i]
                
                # Check for swing high
                if current_price == np.max(window):
                    state["swings"]["highs"].append((len(state["price_buffer"]) - len(prices) + i, current_price))
                
                # Check for swing low
                if current_price == np.min(window):
                    state["swings"]["lows"].append((len(state["price_buffer"]) - len(prices) + i, current_price))
            
            # Limit stored swings
            max_swings = 10
            if len(state["swings"]["highs"]) > max_swings:
                state["swings"]["highs"] = state["swings"]["highs"][-max_swings:]
            if len(state["swings"]["lows"]) > max_swings:
                state["swings"]["lows"] = state["swings"]["lows"][-max_swings:]
            
            # Update pivot point if we have both highs and lows
            if state["swings"]["highs"] and state["swings"]["lows"]:
                latest_high = state["swings"]["highs"][-1]
                latest_low = state["swings"]["lows"][-1]
                
                # Determine pivot and trend
                if latest_high[0] > latest_low[0]:  # Most recent is high
                    state["pivot_point"] = latest_high
                    state["secondary_point"] = latest_low
                    state["current_trend"] = MarketDirection.BEARISH  # Fans drawn down from high
                else:  # Most recent is low
                    state["pivot_point"] = latest_low
                    state["secondary_point"] = latest_high
                    state["current_trend"] = MarketDirection.BULLISH  # Fans drawn up from low
                
                # Calculate fans for each level
                pivot_point = state["pivot_point"]
                secondary_point = state["secondary_point"]
                
                if pivot_point and secondary_point:
                    price_range = abs(pivot_point[1] - secondary_point[1])
                    bar_distance = abs(pivot_point[0] - secondary_point[0])
                    
                    # Save parameters for fan calculation
                    state["fan_values"] = {
                        "pivot_point": pivot_point,
                        "price_range": price_range,
                        "bar_distance": bar_distance,
                        "trend": state["current_trend"]
                    }
        
        return state


class FibonacciTimeZones(AdvancedAnalysisBase):
    """Fibonacci Time Zones calculator
    
    Fibonacci Time Zones are vertical lines based on the Fibonacci sequence,
    projected forward in time from a significant low or high. They identify
    potential areas where price reversals or significant movements may occur.
    """
    
    def __init__(self, 
                 auto_detect_pivot: bool = True,
                 lookback_period: int = 100,
                 price_column: str = "close",
                 projection_count: int = 10,
                 use_extended_sequence: bool = False,
                 pivot_type: str = "auto"
                ):
        """
        Initialize the Fibonacci Time Zones calculator
        
        Args:
            auto_detect_pivot: Automatically detect a significant pivot point
            lookback_period: Max periods to look back for pivot detection
            price_column: Column name for price data
            projection_count: Number of Fibonacci time zones to project
            use_extended_sequence: Whether to use extended Fibonacci sequence
            pivot_type: Force pivot type ('high', 'low', or 'auto')
        """
        parameters = {
            "auto_detect_pivot": auto_detect_pivot,
            "lookback_period": lookback_period,
            "price_column": price_column,
            "projection_count": projection_count,
            "use_extended_sequence": use_extended_sequence,
            "pivot_type": pivot_type
        }
        super().__init__("Fibonacci Time Zones", parameters)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci time zones from a significant pivot point
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Fibonacci time zone markers
        """
        result_df = df.copy()
        
        # Check if we have enough data
        if len(df) < self.parameters["lookback_period"]:
            return result_df
        
        # Generate Fibonacci sequence for time projections
        fib_sequence = self._generate_fibonacci_sequence(self.parameters["projection_count"])
        
        # Use only the lookback period
        analysis_df = df.iloc[-self.parameters["lookback_period"]:]
        price_col = self.parameters["price_column"]
        
        # Find pivot point for time zone calculations
        pivot_idx = None
        if self.parameters["auto_detect_pivot"]:
            # Detect swings
            swing_df = detect_swings(analysis_df, lookback=5, price_col=price_col)
            
            # Find recent swing highs and lows
            swing_highs = swing_df[swing_df["swing_high"]].index
            swing_lows = swing_df[swing_df["swing_low"]].index
            
            if len(swing_highs) > 0 and len(swing_lows) > 0:
                # Determine which pivot to use
                pivot_type = self.parameters["pivot_type"].lower()
                
                if pivot_type == "high" and len(swing_highs) > 0:
                    # Use the most significant high in the lookback period
                    pivot_idx = swing_highs[swing_df.loc[swing_highs][price_col].argmax()]
                elif pivot_type == "low" and len(swing_lows) > 0:
                    # Use the most significant low in the lookback period
                    pivot_idx = swing_lows[swing_df.loc[swing_lows][price_col].argmin()]
                else:  # "auto" or any other value
                    # Find the most significant swing (either high or low)
                    # by comparing the deviation from the mean
                    mean_price = analysis_df[price_col].mean()
                    
                    if len(swing_highs) > 0:
                        max_high = swing_highs[swing_df.loc[swing_highs][price_col].argmax()]
                        max_high_price = swing_df.loc[max_high][price_col]
                        high_deviation = abs(max_high_price - mean_price)
                    else:
                        high_deviation = 0
                    
                    if len(swing_lows) > 0:
                        min_low = swing_lows[swing_df.loc[swing_lows][price_col].argmin()]
                        min_low_price = swing_df.loc[min_low][price_col]
                        low_deviation = abs(min_low_price - mean_price)
                    else:
                        low_deviation = 0
                    
                    # Use the swing with the largest deviation
                    if high_deviation > low_deviation and len(swing_highs) > 0:
                        pivot_idx = max_high
                    elif len(swing_lows) > 0:
                        pivot_idx = min_low
        
        # If we found a pivot or one was provided
        if pivot_idx is not None:
            pivot_loc = df.index.get_loc(pivot_idx)
            
            # Add column for the pivot point
            result_df["fib_time_pivot"] = False
            result_df.loc[pivot_idx, "fib_time_pivot"] = True
            
            # Add columns for each Fibonacci time zone
            for i, fib_value in enumerate(fib_sequence):
                zone_name = f"fib_time_zone_{i+1}"
                result_df[zone_name] = False
                
                # Calculate the bar index for this zone
                zone_loc = pivot_loc + fib_value
                
                # Mark the time zone in the dataframe if it's within range
                if 0 <= zone_loc < len(result_df):
                    result_df.iloc[zone_loc][zone_name] = True
            
            # Add pivot index for reference
            result_df["fib_time_pivot_idx"] = pivot_idx
        
        return result_df
    
    def _generate_fibonacci_sequence(self, count: int) -> List[int]:
        """
        Generate Fibonacci sequence for time zones
        
        Args:
            count: Number of elements in the sequence
            
        Returns:
            List of Fibonacci numbers for time zones
        """
        if self.parameters["use_extended_sequence"]:
            # Extended sequence starting with 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...
            sequence = [0, 1, 1]
            while len(sequence) < count + 2:  # +2 because we skip first two values
                sequence.append(sequence[-1] + sequence[-2])
            
            # Return sequence starting from the third element (skipping 0, 1)
            return sequence[2:count+2]
        else:
            # Standard Fibonacci ratios: 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...
            sequence = [1, 2, 3]
            while len(sequence) < count:
                sequence.append(sequence[-1] + sequence[-2])
            
            return sequence[:count]
    
    def initialize_incremental(self) -> Dict[str, Any]:
        """Initialize state for incremental calculation"""
        return {
            "price_buffer": [],
            "swings": {"highs": [], "lows": []},
            "lookback_period": self.parameters["lookback_period"],
            "price_column": self.parameters["price_column"],
            "projection_count": self.parameters["projection_count"],
            "pivot_idx": None,
            "pivot_type": self.parameters["pivot_type"],
            "time_zones": [],
            "fib_sequence": self._generate_fibonacci_sequence(self.parameters["projection_count"])
        }
    
    def update_incremental(self, state: Dict[str, Any], new_data: Dict[str, float]) -> Dict[str, Any]:
        """Update Fibonacci time zones with new data"""
        # Add new price to buffer
        if self.parameters["price_column"] in new_data:
            price = new_data[self.parameters["price_column"]]
            state["price_buffer"].append(price)
        else:
            return state
            
        # Keep buffer within lookback size
        if len(state["price_buffer"]) > state["lookback_period"]:
            state["price_buffer"] = state["price_buffer"][-state["lookback_period"]:]
        
        # Detect swings only if we don't have a pivot yet
        if state["pivot_idx"] is None and len(state["price_buffer"]) > 10:
            prices = np.array(state["price_buffer"])
            window_size = 5
            
            # Find significant swing points
            for i in range(window_size, len(prices) - window_size):
                window = prices[i-window_size:i+window_size+1]
                current_price = prices[i]
                
                # Check for swing high
                if current_price == np.max(window):
                    state["swings"]["highs"].append((len(state["price_buffer"]) - len(prices) + i, current_price))
                
                # Check for swing low
                if current_price == np.min(window):
                    state["swings"]["lows"].append((len(state["price_buffer"]) - len(prices) + i, current_price))
            
            # Determine pivot point if we have enough swings
            if state["swings"]["highs"] or state["swings"]["lows"]:
                pivot_type = state["pivot_type"].lower()
                
                if pivot_type == "high" and state["swings"]["highs"]:
                    # Find the most significant high
                    highest_idx = max(range(len(state["swings"]["highs"])), 
                                       key=lambda i: state["swings"]["highs"][i][1])
                    state["pivot_idx"] = state["swings"]["highs"][highest_idx][0]
                elif pivot_type == "low" and state["swings"]["lows"]:
                    # Find the most significant low
                    lowest_idx = min(range(len(state["swings"]["lows"])),
                                      key=lambda i: state["swings"]["lows"][i][1])
                    state["pivot_idx"] = state["swings"]["lows"][lowest_idx][0]
                else:  # "auto" or any other value
                    # Use the most significant swing (highest or lowest)
                    if state["swings"]["highs"] and state["swings"]["lows"]:
                        mean_price = np.mean(prices)
                        highest = max(state["swings"]["highs"], key=lambda x: x[1])[1]
                        lowest = min(state["swings"]["lows"], key=lambda x: x[1])[1]
                        
                        if abs(highest - mean_price) > abs(lowest - mean_price):
                            highest_idx = max(range(len(state["swings"]["highs"])),
                                             key=lambda i: state["swings"]["highs"][i][1])
                            state["pivot_idx"] = state["swings"]["highs"][highest_idx][0]
                        else:
                            lowest_idx = min(range(len(state["swings"]["lows"])),
                                            key=lambda i: state["swings"]["lows"][i][1])
                            state["pivot_idx"] = state["swings"]["lows"][lowest_idx][0]
                    elif state["swings"]["highs"]:
                        highest_idx = max(range(len(state["swings"]["highs"])),
                                         key=lambda i: state["swings"]["highs"][i][1])
                        state["pivot_idx"] = state["swings"]["highs"][highest_idx][0]
                    elif state["swings"]["lows"]:
                        lowest_idx = min(range(len(state["swings"]["lows"])),
                                        key=lambda i: state["swings"]["lows"][i][1])
                        state["pivot_idx"] = state["swings"]["lows"][lowest_idx][0]
        
            # Calculate time zones if we have a pivot
            if state["pivot_idx"] is not None:
                state["time_zones"] = [state["pivot_idx"] + fib for fib in state["fib_sequence"]]
        
        return state


class FibonacciAnalyzer(AdvancedAnalysisBase):
    """
    Complete Fibonacci Analysis suite combining multiple Fibonacci tools
    
    This class provides a comprehensive Fibonacci analysis including:
    - Retracements
    - Extensions
    - Arcs
    - Fans
    - Time zones
    """
    
    def __init__(self, tools: List[str] = None, **kwargs):
        """
        Initialize the Fibonacci Analyzer
        
        Args:
            tools: List of Fibonacci tools to include 
                  (options: "retracement", "extension", "arc", "fan", "time")
            **kwargs: Additional parameters for individual tools
        """
        tools = tools or ["retracement", "extension"]
        parameters = {
            "tools": tools,
            **kwargs
        }
        super().__init__("Fibonacci Analyzer", parameters)
        
        # Initialize individual tools
        self.components = {}
        if "retracement" in tools:
            self.components["retracement"] = FibonacciRetracement(**kwargs)
        if "extension" in tools:
            self.components["extension"] = FibonacciExtension(**kwargs)
        if "arc" in tools:
            self.components["arc"] = FibonacciArcs(**kwargs)
        if "fan" in tools:
            self.components["fan"] = FibonacciFans(**kwargs)
        if "time" in tools:
            self.components["time"] = FibonacciTimeZones(**kwargs)
        # Additional Fibonacci tools would be initialized here
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all Fibonacci components
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all Fibonacci analysis results
        """
        result_df = df.copy()
        
        # Process with each component
        for name, component in self.components.items():
            component_df = component.calculate(df)
            
            # Add columns from component_df to result_df
            # (excluding the original OHLCV columns)
            new_columns = [col for col in component_df.columns if col not in df.columns]
            for col in new_columns:
                result_df[col] = component_df[col]
        
        return result_df
    
    def initialize_incremental(self) -> Dict[str, Any]:
        """Initialize state for incremental calculation"""
        state = {}
        
        # Initialize state for each component
        for name, component in self.components.items():
            state[name] = component.initialize_incremental()
            
        return state
    
    def update_incremental(self, state: Dict[str, Any], new_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Update all Fibonacci components incrementally
        
        Args:
            state: Current calculation state
            new_data: New price data point
            
        Returns:
            Updated state with all Fibonacci components
        """
        # Process with each component
        for name, component in self.components.items():
            state[name] = component.update_incremental(state[name], new_data)
            
        return state
