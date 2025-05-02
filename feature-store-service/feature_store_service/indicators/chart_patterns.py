"""
Advanced Chart Patterns Module.

This module provides implementations of common chart patterns for automated recognition.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from enum import Enum

from feature_store_service.indicators.base_indicator import BaseIndicator


class PatternType(Enum):
    """Enum representing different pattern types."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    CONTINUATION = "continuation"
    REVERSAL = "reversal"
    CONSOLIDATION = "consolidation"
    UNDEFINED = "undefined"


class ChartPatternRecognizer(BaseIndicator):
    """
    Chart Pattern Recognizer
    
    Identifies common chart patterns like Head and Shoulders, Double Tops/Bottoms,
    Triangle patterns, Flag patterns, etc.
    """
    
    category = "pattern"
    
    def __init__(
        self, 
        lookback_period: int = 100, 
        pattern_types: Optional[List[str]] = None,
        min_pattern_size: int = 10,
        max_pattern_size: int = 50,
        sensitivity: float = 0.75,
        **kwargs
    ):
        """
        Initialize Chart Pattern Recognizer.
        
        Args:
            lookback_period: Number of bars to look back for pattern recognition
            pattern_types: List of pattern types to look for (None = all patterns)
            min_pattern_size: Minimum size of patterns to recognize (in bars)
            max_pattern_size: Maximum size of patterns to recognize (in bars)
            sensitivity: Sensitivity of pattern detection (0.0-1.0)
            **kwargs: Additional parameters
        """
        self.lookback_period = lookback_period
        self.min_pattern_size = min_pattern_size
        self.max_pattern_size = max_pattern_size
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        
        # Set pattern types to recognize
        all_patterns = [
            "head_and_shoulders", "inverse_head_and_shoulders",
            "double_top", "double_bottom",
            "triple_top", "triple_bottom",
            "ascending_triangle", "descending_triangle", "symmetric_triangle",
            "flag", "pennant",
            "wedge_rising", "wedge_falling",
            "rectangle"
        ]
        
        if pattern_types is None:
            self.pattern_types = all_patterns
        else:
            self.pattern_types = [p for p in pattern_types if p in all_patterns]
            
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate chart pattern recognition for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with pattern recognition values
        """
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Initialize pattern columns with zeros
        for pattern in self.pattern_types:
            result[f"pattern_{pattern}"] = 0
            
        # Look for each requested pattern type
        if "head_and_shoulders" in self.pattern_types:
            result = self._find_head_and_shoulders(result, inverse=False)
            
        if "inverse_head_and_shoulders" in self.pattern_types:
            result = self._find_head_and_shoulders(result, inverse=True)
            
        if "double_top" in self.pattern_types:
            result = self._find_double_formation(result, is_top=True)
            
        if "double_bottom" in self.pattern_types:
            result = self._find_double_formation(result, is_top=False)
            
        if "triple_top" in self.pattern_types:
            result = self._find_triple_formation(result, is_top=True)
            
        if "triple_bottom" in self.pattern_types:
            result = self._find_triple_formation(result, is_top=False)
            
        if any(p in self.pattern_types for p in ["ascending_triangle", "descending_triangle", "symmetric_triangle"]):
            result = self._find_triangle_patterns(result)
            
        if any(p in self.pattern_types for p in ["flag", "pennant"]):
            result = self._find_flag_pennant_patterns(result)
            
        if any(p in self.pattern_types for p in ["wedge_rising", "wedge_falling"]):
            result = self._find_wedge_patterns(result)
            
        if "rectangle" in self.pattern_types:
            result = self._find_rectangle_pattern(result)
            
        # Add a consolidated patterns column for easy filtering
        result["has_pattern"] = (result[[f"pattern_{p}" for p in self.pattern_types]].sum(axis=1) > 0).astype(int)
        
        # Add pattern strength metric
        self._calculate_pattern_strength(result)
        
        return result
        
    def _find_head_and_shoulders(self, data: pd.DataFrame, inverse: bool = False) -> pd.DataFrame:
        \"\"\"
        Find Head and Shoulders or Inverse Head and Shoulders patterns.
        
        Args:
            data: DataFrame with price data
            inverse: If True, look for inverse head and shoulders (bullish)
        
        Returns:
            DataFrame with pattern recognition markers
        \"\"\"
        result = data.copy()
        pattern_name = "inverse_head_and_shoulders" if inverse else "head_and_shoulders"
        
        # Use appropriate price series based on pattern type
        price_series = result['low'] if inverse else result['high']
        
        # Initialize pattern column if not already done
        if f"pattern_{pattern_name}" not in result.columns:
            result[f"pattern_{pattern_name}"] = 0
            
        # We need at least min_pattern_size bars for a valid pattern
        if len(result) < self.min_pattern_size:
            return result
            
        # Get the lookback data
        lookback_len = min(len(result), self.lookback_period)
        lookback_data = result.iloc[-lookback_len:]
        
        # Find peaks or troughs using a more robust method if available, e.g., scipy.signal.find_peaks
        # For simplicity, using the existing _find_peaks_troughs
        peaks_or_troughs_indices = self._find_peaks_troughs(lookback_data, find_peaks=not inverse)
        
        # Convert relative indices to absolute indices of the original DataFrame
        absolute_indices = [lookback_data.index[idx] for idx in peaks_or_troughs_indices]

        # We need at least 5 peaks/troughs for a head and shoulders pattern
        if len(absolute_indices) < 5:
            return result
            
        # Check each window of 5 peaks/troughs for the pattern
        for i in range(len(absolute_indices) - 4):
            points_indices = absolute_indices[i:i+5]
            
            # Extract points indices for left shoulder, head, right shoulder
            left_shoulder_idx = points_indices[0]
            head_idx = points_indices[2]
            right_shoulder_idx = points_indices[4]
            
            # Extract heights (y-values) from the original result DataFrame
            ls_height = price_series.loc[left_shoulder_idx]
            head_height = price_series.loc[head_idx] 
            rs_height = price_series.loc[right_shoulder_idx]
            
            # For a valid pattern:
            # 1. Head should be higher (lower for inverse) than both shoulders
            # 2. Shoulders should be roughly at the same level
            # 3. Pattern should form within a reasonable timeframe
            
            head_higher = (
                (head_height > ls_height and head_height > rs_height) if not inverse 
                else (head_height < ls_height and head_height < rs_height)
            )
            
            # Check if shoulders are at similar levels (within 10% * sensitivity)
            # Avoid division by zero if ls_height is 0
            if ls_height == 0: continue
            shoulder_diff_pct = abs(ls_height - rs_height) / ls_height
            shoulders_similar = shoulder_diff_pct < (0.1 / max(self.sensitivity, 0.1)) # Ensure sensitivity is not zero
            
            # Check if pattern is within reasonable size constraints (using absolute indices)
            pattern_width = result.index.get_loc(right_shoulder_idx) - result.index.get_loc(left_shoulder_idx)
            reasonable_size = self.min_pattern_size <= pattern_width <= self.max_pattern_size
            
            if head_higher and shoulders_similar and reasonable_size:
                # Calculate neckline from the troughs/peaks between shoulders and head
                neckline_left_idx = points_indices[1]
                neckline_right_idx = points_indices[3]
                
                # Use the opposite price series for neckline points
                neckline_price_series = result['high'] if inverse else result['low']
                neckline_left_price = neckline_price_series.loc[neckline_left_idx]
                neckline_right_price = neckline_price_series.loc[neckline_right_idx]

                # Calculate neckline slope and intercept using absolute indices
                neckline_left_loc = result.index.get_loc(neckline_left_idx)
                neckline_right_loc = result.index.get_loc(neckline_right_idx)
                
                if neckline_right_loc == neckline_left_loc: continue # Avoid division by zero

                slope = (neckline_right_price - neckline_left_price) / (neckline_right_loc - neckline_left_loc)
                intercept = neckline_left_price - (slope * neckline_left_loc)

                # Mark the pattern in the result DataFrame using absolute indices
                start_loc = result.index.get_loc(left_shoulder_idx)
                end_loc = result.index.get_loc(right_shoulder_idx)
                result.iloc[start_loc:end_loc + 1, result.columns.get_loc(f"pattern_{pattern_name}")] = 1
                        
                # Add neckline information
                neckline_col = f"pattern_{pattern_name}_neckline"
                if neckline_col not in result.columns:
                    result[neckline_col] = np.nan # Use NaN for missing values
                    
                # Project the neckline
                projection_end_loc = min(len(result), end_loc + self.projection_bars)
                for loc in range(start_loc, projection_end_loc):
                    neckline_val = (slope * loc) + intercept
                    result.iloc[loc, result.columns.get_loc(neckline_col)] = neckline_val
                    
        return result
        
    def _find_double_formation(self, data: pd.DataFrame, is_top: bool) -> pd.DataFrame:
        \"\"\"
        Find Double Top or Double Bottom patterns.
        
        Args:
            data: DataFrame with price data
            is_top: If True, look for double top (bearish), otherwise double bottom (bullish)
        
        Returns:
            DataFrame with pattern recognition markers
        \"\"\"
        result = data.copy()
        pattern_name = "double_top" if is_top else "double_bottom"
        
        # Use appropriate price series based on pattern type
        price_series = result['high'] if is_top else result['low']
        
        # Initialize pattern column if not already done
        if f"pattern_{pattern_name}" not in result.columns:
            result[f"pattern_{pattern_name}"] = 0
            
        # We need at least min_pattern_size bars for a valid pattern
        if len(result) < self.min_pattern_size:
            return result
            
        # Get the lookback data
        lookback_len = min(len(result), self.lookback_period)
        lookback_data = result.iloc[-lookback_len:]
        
        # Find peaks or troughs
        peaks_or_troughs_indices = self._find_peaks_troughs(lookback_data, find_peaks=is_top)
        absolute_indices = [lookback_data.index[idx] for idx in peaks_or_troughs_indices]

        # We need at least 2 peaks/troughs for a double formation
        if len(absolute_indices) < 2:
            return result
            
        # Check each consecutive pair of peaks/troughs
        for i in range(len(absolute_indices) - 1):
            first_abs_idx = absolute_indices[i]
            second_abs_idx = absolute_indices[i+1]
            
            # Extract heights (y-values) from the original result DataFrame
            first_height = price_series.loc[first_abs_idx]
            second_height = price_series.loc[second_abs_idx]
            
            # For a valid pattern:
            # 1. The two peaks/troughs should be at similar levels
            # 2. Pattern should form within a reasonable timeframe
            # 3. There should be a significant dip/rise between them
            
            # Check if heights are similar (within 5% * sensitivity)
            if first_height == 0: continue # Avoid division by zero
            height_diff_pct = abs(first_height - second_height) / first_height
            similar_heights = height_diff_pct < (0.05 / max(self.sensitivity, 0.1))
            
            # Check if pattern is within reasonable size constraints (using absolute indices)
            first_loc = result.index.get_loc(first_abs_idx)
            second_loc = result.index.get_loc(second_abs_idx)
            pattern_width = second_loc - first_loc
            reasonable_size = self.min_pattern_size <= pattern_width <= self.max_pattern_size
            
            # Find the middle point (highest or lowest between the two peaks/troughs)
            middle_slice = result.loc[first_abs_idx:second_abs_idx] # Slice using absolute indices
            if middle_slice.empty or len(middle_slice) < 3: continue # Need points between peaks/troughs

            if is_top:
                middle_idx = middle_slice['low'][1:-1].idxmin() # Exclude the peaks themselves
                middle_val = result.loc[middle_idx, 'low']
                # Calculate how deep the middle point is relative to the tops
                avg_top = (first_height + second_height) / 2
                if avg_top == 0: continue
                depth_pct = (avg_top - middle_val) / avg_top
            else:
                middle_idx = middle_slice['high'][1:-1].idxmax() # Exclude the troughs themselves
                middle_val = result.loc[middle_idx, 'high']
                # Calculate how high the middle point is relative to the bottoms
                avg_bottom = (first_height + second_height) / 2
                if avg_bottom == 0: continue
                depth_pct = (middle_val - avg_bottom) / avg_bottom
            
            # Check if the middle point is deep/high enough
            significant_middle = depth_pct > (0.10 * max(self.sensitivity, 0.1))
            
            if similar_heights and reasonable_size and significant_middle:
                # Mark the pattern in the result DataFrame using absolute locations
                result.iloc[first_loc:second_loc + 1, result.columns.get_loc(f"pattern_{pattern_name}")] = 1
                
                # Add neckline (support or resistance level)
                neckline_col = f"pattern_{pattern_name}_neckline"
                if neckline_col not in result.columns:
                    result[neckline_col] = np.nan # Use NaN for missing values
                    
                # Use the middle point to establish the neckline
                neckline_val = middle_val
                projection_end_loc = min(len(result), second_loc + self.projection_bars)
                result.iloc[first_loc:projection_end_loc, result.columns.get_loc(neckline_col)] = neckline_val
        
        return result
        
    def _find_triple_formation(self, data: pd.DataFrame, is_top: bool) -> pd.DataFrame:
        \"\"\"
        Find Triple Top or Triple Bottom patterns.
        
        Args:
            data: DataFrame with price data
            is_top: If True, look for triple top (bearish), otherwise triple bottom (bullish)
        
        Returns:
            DataFrame with pattern recognition markers
        \"\"\"
        result = data.copy()
        pattern_name = "triple_top" if is_top else "triple_bottom"
        
        # Use appropriate price series based on pattern type
        price_series = result['high'] if is_top else result['low']
        
        # Initialize pattern column if not already done
        if f"pattern_{pattern_name}" not in result.columns:
            result[f"pattern_{pattern_name}"] = 0
            
        # We need at least min_pattern_size bars for a valid pattern
        if len(result) < self.min_pattern_size:
            return result
            
        # Get the lookback data
        lookback_len = min(len(result), self.lookback_period)
        lookback_data = result.iloc[-lookback_len:]
        
        # Find peaks or troughs
        peaks_or_troughs_indices = self._find_peaks_troughs(lookback_data, find_peaks=is_top)
        absolute_indices = [lookback_data.index[idx] for idx in peaks_or_troughs_indices]

        # We need at least 3 peaks/troughs for a triple formation
        if len(absolute_indices) < 3:
            return result
            
        # Check each consecutive triplet of peaks/troughs
        for i in range(len(absolute_indices) - 2):
            first_abs_idx = absolute_indices[i]
            second_abs_idx = absolute_indices[i+1]
            third_abs_idx = absolute_indices[i+2]
            
            # Extract heights (y-values) from the original result DataFrame
            first_height = price_series.loc[first_abs_idx]
            second_height = price_series.loc[second_abs_idx]
            third_height = price_series.loc[third_abs_idx]
            
            # For a valid pattern:
            # 1. All three peaks/troughs should be at similar levels
            # 2. Pattern should form within a reasonable timeframe
            
            # Check if heights are similar (within 5% * sensitivity)
            heights = [first_height, second_height, third_height]
            max_height = max(heights)
            min_height = min(heights)
            if max_height == 0: continue # Avoid division by zero
            height_diff_pct = (max_height - min_height) / max_height
            similar_heights = height_diff_pct < (0.05 / max(self.sensitivity, 0.1))
            
            # Check if pattern is within reasonable size constraints (using absolute indices)
            first_loc = result.index.get_loc(first_abs_idx)
            third_loc = result.index.get_loc(third_abs_idx)
            pattern_width = third_loc - first_loc
            reasonable_size = self.min_pattern_size <= pattern_width <= self.max_pattern_size
            
            if similar_heights and reasonable_size:
                # Mark the pattern in the result DataFrame using absolute locations
                result.iloc[first_loc:third_loc + 1, result.columns.get_loc(f"pattern_{pattern_name}")] = 1
                
                # Add neckline (support or resistance level)
                neckline_col = f"pattern_{pattern_name}_neckline"
                if neckline_col not in result.columns:
                    result[neckline_col] = np.nan # Use NaN for missing values
                
                # Find the two lowest/highest points between the three peaks/troughs
                slice1 = result.loc[first_abs_idx:second_abs_idx]
                slice2 = result.loc[second_abs_idx:third_abs_idx]
                if slice1.empty or len(slice1) < 3 or slice2.empty or len(slice2) < 3: continue

                if is_top:
                    middle_idx = slice1['low'][1:-1].idxmin()
                    middle_val = result.loc[middle_idx, 'low']
                    # Calculate how deep the middle point is relative to the tops
                    avg_top = (first_height + second_height) / 2
                    if avg_top == 0: continue
                    depth_pct = (avg_top - middle_val) / avg_top
                else:
                    middle_idx = slice2['high'][1:-1].idxmax()
                    middle_val = result.loc[middle_idx, 'high']
                    # Calculate how high the middle point is relative to the bottoms
                    avg_bottom = (first_height + second_height) / 2
                    if avg_bottom == 0: continue
                    depth_pct = (middle_val - avg_bottom) / avg_bottom
                
                # Check if the middle point is deep/high enough
                significant_middle = depth_pct > (0.10 * max(self.sensitivity, 0.1))
                
                if similar_heights and reasonable_size and significant_middle:
                    # Mark the pattern in the result DataFrame using absolute locations
                    result.iloc[first_loc:third_loc + 1, result.columns.get_loc(f"pattern_{pattern_name}")] = 1
                    
                    # Add neckline (support or resistance level)
                    neckline_col = f"pattern_{pattern_name}_neckline"
                    if neckline_col not in result.columns:
                        result[neckline_col] = np.nan # Use NaN for missing values
                        
                    # Use the middle point to establish the neckline
                    neckline_val = middle_val
                    projection_end_loc = min(len(result), third_loc + self.projection_bars)
                    result.iloc[first_loc:projection_end_loc, result.columns.get_loc(neckline_col)] = neckline_val
        
        return result
        
    def _find_triangle_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Find Triangle patterns (Ascending, Descending, Symmetric).
        
        Args:
            data: DataFrame with price data
        
        Returns:
            DataFrame with triangle pattern recognition markers
        \"\"\"
        result = data.copy()
        
        # We need at least min_pattern_size bars for a valid pattern
        if len(result) < self.min_pattern_size:
            return result
            
        # Initialize pattern columns if not already done
        triangle_types = ["ascending_triangle", "descending_triangle", "symmetric_triangle"]
        for pattern in triangle_types:
            if f"pattern_{pattern}" not in result.columns:
                result[f"pattern_{pattern}"] = 0
        
        # Iterate through potential pattern end points
        for end_loc in range(self.min_pattern_size, len(result)):
            start_loc = max(0, end_loc - self.lookback_period)
            lookback_data = result.iloc[start_loc:end_loc]

            if len(lookback_data) < self.min_pattern_size: continue

            # Find peaks and troughs within the lookback window
            peak_indices_rel = self._find_peaks_troughs(lookback_data, find_peaks=True)
            trough_indices_rel = self._find_peaks_troughs(lookback_data, find_peaks=False)

            # Convert relative indices to absolute locations
            peak_locs = [start_loc + idx for idx in peak_indices_rel]
            trough_locs = [start_loc + idx for idx in trough_indices_rel]

            # We need at least 2 peaks and 2 troughs to form a triangle
            if len(peak_locs) < 2 or len(trough_locs) < 2:
                continue

            # Calculate trend lines for highs and lows using absolute locations
            high_slope, high_intercept = self._calculate_trendline_loc(result, peak_locs, use_high=True)
            low_slope, low_intercept = self._calculate_trendline_loc(result, trough_locs, use_high=False)

            # Define slope thresholds based on sensitivity
            flat_threshold = 0.001 * (1 / max(self.sensitivity, 0.1)) # Smaller threshold for higher sensitivity
            significant_slope_threshold = 0.002 * max(self.sensitivity, 0.1) # Larger threshold for higher sensitivity

            # Triangle pattern types are determined by the slopes of the trend lines
            is_ascending = abs(high_slope) < flat_threshold and low_slope > significant_slope_threshold
            is_descending = abs(low_slope) < flat_threshold and high_slope < -significant_slope_threshold
            is_symmetric = high_slope < -significant_slope_threshold and low_slope > significant_slope_threshold

            # Determine triangle type and mark it on the chart
            triangle_type = None
            if is_ascending and "ascending_triangle" in self.pattern_types:
                triangle_type = "ascending_triangle"
            elif is_descending and "descending_triangle" in self.pattern_types:
                triangle_type = "descending_triangle"
            elif is_symmetric and "symmetric_triangle" in self.pattern_types:
                triangle_type = "symmetric_triangle"
            
            if triangle_type:
                # Find the start and end locations of the pattern within the lookback
                pattern_start_loc = min(min(peak_locs), min(trough_locs))
                pattern_end_loc = max(max(peak_locs), max(trough_locs))
                
                # Check if pattern is within reasonable size constraints
                pattern_width = pattern_end_loc - pattern_start_loc
                if self.min_pattern_size <= pattern_width <= self.max_pattern_size:
                    # Mark the pattern in the result DataFrame
                    result.iloc[pattern_start_loc:pattern_end_loc + 1, result.columns.get_loc(f"pattern_{triangle_type}")] = 1
                    
                    # Add upper and lower trendlines
                    upper_trendline_col = f"pattern_{triangle_type}_upper"
                    lower_trendline_col = f"pattern_{triangle_type}_lower"
                    
                    if upper_trendline_col not in result.columns:
                        result[upper_trendline_col] = np.nan
                    if lower_trendline_col not in result.columns:
                        result[lower_trendline_col] = np.nan
                    
                    # Calculate trendline values for projection
                    projection_end_loc = min(len(result), pattern_end_loc + self.projection_bars)
                    for loc in range(pattern_start_loc, projection_end_loc):
                        upper_val = (high_slope * loc) + high_intercept
                        lower_val = (low_slope * loc) + low_intercept
                        
                        result.iloc[loc, result.columns.get_loc(upper_trendline_col)] = upper_val
                        result.iloc[loc, result.columns.get_loc(lower_trendline_col)] = lower_val
        
        return result
        
    def _find_flag_pennant_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Find Flag and Pennant patterns. Requires 'volume' column.
        
        Args:
            data: DataFrame with price data (must include 'volume')
        
        Returns:
            DataFrame with flag/pennant pattern recognition markers
        \"\"\"
        result = data.copy()
        
        if 'volume' not in result.columns:
            # If volume is missing, cannot reliably detect flags/pennants
            # logger.warning("Volume column missing, cannot detect Flag/Pennant patterns.")
            return result 

        # Initialize pattern columns if not already done
        for pattern in ["flag", "pennant"]:
            if f"pattern_{pattern}" not in result.columns:
                result[f"pattern_{pattern}"] = 0
                
        # We need at least min_pattern_size bars for a valid pattern
        if len(result) < self.min_pattern_size:
            return result
            
        # Parameters
        flagpole_min_bars = max(3, int(self.min_pattern_size * 0.2)) # Min bars for flagpole
        flagpole_min_change_pct = 0.03 * max(self.sensitivity, 0.1) # Min price change for flagpole
        consolidation_min_bars = max(5, int(self.min_pattern_size * 0.5)) # Min bars for consolidation
        consolidation_max_bars = self.max_pattern_size - flagpole_min_bars
        parallel_threshold = 0.005 * (1 / max(self.sensitivity, 0.1)) # Threshold for parallel lines (flag)
        converging_threshold = 0.005 * max(self.sensitivity, 0.1) # Threshold for converging lines (pennant)
        volume_decrease_factor = 0.5 # Volume should decrease during consolidation

        # Iterate through potential pattern end points
        for end_loc in range(self.min_pattern_size, len(result)):
            # Look back for potential flagpole + consolidation
            max_lookback = flagpole_min_bars + consolidation_max_bars
            start_loc = max(0, end_loc - max_lookback)
            lookback_data = result.iloc[start_loc:end_loc]

            if len(lookback_data) < self.min_pattern_size: continue

            # 1. Identify potential flagpole
            for fp_end_loc_rel in range(flagpole_min_bars, len(lookback_data) - consolidation_min_bars):
                fp_start_loc_rel = 0
                flagpole_data = lookback_data.iloc[fp_start_loc_rel:fp_end_loc_rel]
                
                if len(flagpole_data) < flagpole_min_bars: continue

                fp_start_price = flagpole_data['close'].iloc[0]
                fp_end_price = flagpole_data['close'].iloc[-1]
                if fp_start_price == 0: continue

                price_change_pct = (fp_end_price - fp_start_price) / fp_start_price
                is_strong_move = abs(price_change_pct) >= flagpole_min_change_pct
                
                if not is_strong_move: continue

                is_uptrend_pole = price_change_pct > 0
                
                # 2. Look for consolidation after the flagpole
                cons_start_loc_rel = fp_end_loc_rel
                cons_end_loc_rel = len(lookback_data) # End of current lookback window
                consolidation_data = lookback_data.iloc[cons_start_loc_rel:cons_end_loc_rel]

                if len(consolidation_data) < consolidation_min_bars: continue

                # Check volume decrease during consolidation
                avg_volume_pole = flagpole_data['volume'].mean()
                avg_volume_cons = consolidation_data['volume'].mean()
                if avg_volume_pole == 0 or avg_volume_cons / avg_volume_pole > volume_decrease_factor:
                    continue # Volume did not decrease significantly

                # Calculate upper and lower trendlines of consolidation using relative locations
                highs = consolidation_data['high']
                lows = consolidation_data['low']
                x_range = np.arange(len(consolidation_data))

                try:
                    high_slope, high_intercept = np.polyfit(x_range, highs, 1)
                    low_slope, low_intercept = np.polyfit(x_range, lows, 1)
                except (np.linalg.LinAlgError, ValueError):
                    continue # Could not fit lines

                # Check for Flag: Parallel trendlines, sloping against the trend
                is_parallel = abs(high_slope - low_slope) < parallel_threshold
                is_counter_trend = (is_uptrend_pole and high_slope < -parallel_threshold / 2) or \
                                   (not is_uptrend_pole and high_slope > parallel_threshold / 2)
                is_flag = is_parallel and is_counter_trend

                # Check for Pennant: Converging trendlines
                is_converging = (high_slope < -converging_threshold and low_slope > converging_threshold) or \
                                (high_slope > converging_threshold and low_slope < -converging_threshold) # Allow both directions
                is_pennant = is_converging

                pattern_type = None
                if is_flag and "flag" in self.pattern_types:
                    pattern_type = "flag"
                elif is_pennant and "pennant" in self.pattern_types:
                    pattern_type = "pennant"

                if pattern_type:
                    # Mark pattern using absolute locations
                    pattern_start_loc_abs = start_loc + fp_start_loc_rel
                    pattern_end_loc_abs = start_loc + cons_end_loc_rel - 1 # End loc is inclusive

                    result.iloc[pattern_start_loc_abs:pattern_end_loc_abs + 1, result.columns.get_loc(f"pattern_{pattern_type}")] = 1
                    # Potentially break here if we only want the first match ending at end_loc
                    # break # Found a pattern ending here

        return result
        
    def _find_wedge_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Find Rising and Falling Wedge patterns.
        
        Args:
            data: DataFrame with price data
        
        Returns:
            DataFrame with wedge pattern recognition markers
        \"\"\"
        result = data.copy()
        
        # Initialize pattern columns if not already done
        for pattern in ["wedge_rising", "wedge_falling"]:
            if f"pattern_{pattern}" not in result.columns:
                result[f"pattern_{pattern}"] = 0
                
        # We need at least min_pattern_size bars for a valid pattern
        if len(result) < self.min_pattern_size:
            return result
            
        # Iterate through potential pattern end points
        for end_loc in range(self.min_pattern_size, len(result)):
            start_loc = max(0, end_loc - self.lookback_period)
            lookback_data = result.iloc[start_loc:end_loc]

            if len(lookback_data) < self.min_pattern_size: continue

            # Find peaks and troughs within the lookback window
            peak_indices_rel = self._find_peaks_troughs(lookback_data, find_peaks=True)
            trough_indices_rel = self._find_peaks_troughs(lookback_data, find_peaks=False)

            # Convert relative indices to absolute locations
            peak_locs = [start_loc + idx for idx in peak_indices_rel]
            trough_locs = [start_loc + idx for idx in trough_indices_rel]

            # We need at least 2 peaks and 2 troughs to form a wedge
            if len(peak_locs) < 2 or len(trough_locs) < 2:
                continue
                
            # Calculate trend lines for highs and lows using absolute locations
            high_slope, high_intercept = self._calculate_trendline_loc(result, peak_locs, use_high=True)
            low_slope, low_intercept = self._calculate_trendline_loc(result, trough_locs, use_high=False)
            
            # Define slope thresholds based on sensitivity
            slope_threshold = 0.002 * max(self.sensitivity, 0.1)

            # Rising wedge: Both trendlines slope upward, converging (low_slope > high_slope)
            is_rising_wedge = (high_slope > slope_threshold and low_slope > slope_threshold and low_slope > high_slope)
            
            # Falling wedge: Both trendlines slope downward, converging (high_slope > low_slope)
            is_falling_wedge = (high_slope < -slope_threshold and low_slope < -slope_threshold and high_slope > low_slope)
            
            pattern_type = None
            if is_rising_wedge and "wedge_rising" in self.pattern_types:
                pattern_type = "wedge_rising"
            elif is_falling_wedge and "wedge_falling" in self.pattern_types:
                pattern_type = "wedge_falling"
            
            if pattern_type:
                # Find the start and end locations of the pattern
                pattern_start_loc = min(min(peak_locs), min(trough_locs))
                pattern_end_loc = max(max(peak_locs), max(trough_locs))
                
                # Check if pattern is within reasonable size constraints
                pattern_width = pattern_end_loc - pattern_start_loc
                if self.min_pattern_size <= pattern_width <= self.max_pattern_size:
                    # Mark the pattern in the result DataFrame
                    result.iloc[pattern_start_loc:pattern_end_loc + 1, result.columns.get_loc(f"pattern_{pattern_type}")] = 1
            
                    # Add upper and lower trendlines
                    upper_trendline_col = f"pattern_{pattern_type}_upper"
                    lower_trendline_col = f"pattern_{pattern_type}_lower"
                    
                    if upper_trendline_col not in result.columns:
                        result[upper_trendline_col] = np.nan
                    if lower_trendline_col not in result.columns:
                        result[lower_trendline_col] = np.nan
                    
                    # Calculate trendline values for projection
                    projection_end_loc = min(len(result), pattern_end_loc + self.projection_bars)
                    for loc in range(pattern_start_loc, projection_end_loc):
                        upper_val = (high_slope * loc) + high_intercept
                        lower_val = (low_slope * loc) + low_intercept
                        
                        result.iloc[loc, result.columns.get_loc(upper_trendline_col)] = upper_val
                        result.iloc[loc, result.columns.get_loc(lower_trendline_col)] = lower_val
        
        return result
        
    def _find_rectangle_pattern(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Find Rectangle pattern (horizontal support and resistance channels).
        
        Args:
            data: DataFrame with price data
        
        Returns:
            DataFrame with rectangle pattern recognition markers
        \"\"\"
        result = data.copy()
        pattern_name = "rectangle"
        
        # Initialize pattern column if not already done
        if f"pattern_{pattern_name}" not in result.columns:
            result[f"pattern_{pattern_name}"] = 0
            
        # We need at least min_pattern_size bars for a valid pattern
        if len(result) < self.min_pattern_size:
            return result
            
        # Iterate through potential pattern end points
        for end_loc in range(self.min_pattern_size, len(result)):
            start_loc = max(0, end_loc - self.lookback_period)
            lookback_data = result.iloc[start_loc:end_loc]

            if len(lookback_data) < self.min_pattern_size: continue

            # Find peaks and troughs within the lookback window
            peak_indices_rel = self._find_peaks_troughs(lookback_data, find_peaks=True)
            trough_indices_rel = self._find_peaks_troughs(lookback_data, find_peaks=False)

            # Convert relative indices to absolute locations
            peak_locs = [start_loc + idx for idx in peak_indices_rel]
            trough_locs = [start_loc + idx for idx in trough_indices_rel]

            # We need at least 2 peaks and 2 troughs to form a rectangle
            if len(peak_locs) < 2 or len(trough_locs) < 2:
                continue
                
            # Check for horizontal support and resistance using absolute locations
            resistance_line = self._horizontal_line_fit_loc(result, peak_locs, is_resistance=True)
            support_line = self._horizontal_line_fit_loc(result, trough_locs, is_resistance=False)
            
            if resistance_line is None or support_line is None or resistance_line <= support_line:
                continue # Not a valid channel or lines crossed
                
            # Calculate channel height as percentage of average price
            pattern_start_loc = min(min(peak_locs), min(trough_locs))
            pattern_end_loc = max(max(peak_locs), max(trough_locs))
            avg_price = result['close'].iloc[pattern_start_loc:pattern_end_loc+1].mean()
            if avg_price == 0: continue
            channel_height_pct = (resistance_line - support_line) / avg_price
            
            # Rectangle should have a reasonable channel height (e.g., 1-10% of price, adjusted by sensitivity)
            min_height_pct = 0.01 * max(self.sensitivity, 0.1)
            max_height_pct = 0.10 * (1 / max(self.sensitivity, 0.1))
            reasonable_height = (min_height_pct <= channel_height_pct <= max_height_pct)
            
            # Check if pattern is within reasonable size constraints
            pattern_width = pattern_end_loc - pattern_start_loc
            reasonable_size = self.min_pattern_size <= pattern_width <= self.max_pattern_size
            
            if reasonable_height and reasonable_size:
                # Mark the pattern in the result DataFrame
                result.iloc[pattern_start_loc:pattern_end_loc + 1, result.columns.get_loc(f"pattern_{pattern_name}")] = 1
                
                # Add support and resistance lines
                support_col = f"pattern_{pattern_name}_support"
                resistance_col = f"pattern_{pattern_name}_resistance"
                
                if support_col not in result.columns:
                    result[support_col] = np.nan
                if resistance_col not in result.columns:
                    result[resistance_col] = np.nan
                
                # Set support and resistance values for projection
                projection_end_loc = min(len(result), pattern_end_loc + self.projection_bars)
                result.iloc[pattern_start_loc:projection_end_loc, result.columns.get_loc(support_col)] = support_line
                result.iloc[pattern_start_loc:projection_end_loc, result.columns.get_loc(resistance_col)] = resistance_line
        
        return result
        
    def _calculate_pattern_strength(self, data: pd.DataFrame) -> None:
        \"\"\"
        Calculate and add pattern strength metrics to the DataFrame.
        
        Args:
            data: DataFrame with pattern recognition columns
        \"\"\"
        # Get all pattern columns (excluding neckline/support/resistance)
        pattern_cols = [col for col in data.columns if col.startswith("pattern_") and 
                        col.count("_") == 1 and 
                        "neckline" not in col and 
                        "support" not in col and 
                        "resistance" not in col and
                        "upper" not in col and
                        "lower" not in col]
        
        # Skip if no patterns found or strength column exists
        if not pattern_cols or "pattern_strength" in data.columns:
             # Initialize strength column if it doesn't exist
            if "pattern_strength" not in data.columns:
                data["pattern_strength"] = 0
            # return # Already calculated or no patterns to calculate for

        # Initialize strength column
        data["pattern_strength"] = 0
        
        # Calculate strength for each pattern type found
        for col in pattern_cols:
            # Find contiguous pattern regions using absolute locations
            pattern_regions = self._find_contiguous_regions_loc(data[col])
            
            for start_loc, end_loc in pattern_regions:
                if end_loc < start_loc: continue # Skip invalid regions

                # Calculate pattern strength based on various factors
                length = end_loc - start_loc + 1
                
                # Ensure indices are valid before slicing
                if start_loc < 0 or end_loc >= len(data): continue

                pattern_slice = data.iloc[start_loc:end_loc+1]
                if pattern_slice.empty: continue

                price_range = pattern_slice['high'].max() - pattern_slice['low'].min()
                avg_price = pattern_slice['close'].mean()
                
                # Calculate volume increase (handle potential missing volume or start index)
                volume_increase = 1.0
                if 'volume' in data.columns:
                    prev_volume_start = max(0, start_loc - 10)
                    if start_loc > prev_volume_start: # Ensure there's a previous period
                        prev_volume_slice = data['volume'].iloc[prev_volume_start:start_loc]
                        current_volume_slice = pattern_slice['volume']
                        if not prev_volume_slice.empty and prev_volume_slice.mean() != 0:
                            volume_increase = current_volume_slice.mean() / prev_volume_slice.mean()
                        elif current_volume_slice.mean() > 0: # Handle case where previous volume is zero
                             volume_increase = 2.0 # Assign a default high increase factor
                
                # Normalize factors (avoid division by zero)
                normalized_length = min(1.0, length / self.max_pattern_size) if self.max_pattern_size > 0 else 0
                normalized_range = min(1.0, price_range / (avg_price * 0.1)) if avg_price > 0 else 0
                normalized_volume = min(1.0, max(0, volume_increase - 1)) # Strength from volume *increase*

                # Calculate strength (0-100) - weighted average
                # Weights can be adjusted based on importance
                length_weight = 0.4
                range_weight = 0.4
                volume_weight = 0.2
                
                pattern_strength = int(
                    (normalized_length * length_weight + 
                     normalized_range * range_weight + 
                     normalized_volume * volume_weight) * 100
                )
                
                # Update strength in the DataFrame, taking the max if multiple patterns overlap
                current_strength = data["pattern_strength"].iloc[start_loc:end_loc+1]
                data.iloc[start_loc:end_loc+1, data.columns.get_loc("pattern_strength")] = np.maximum(current_strength, pattern_strength)
    
    def _find_peaks_troughs(self, data: pd.DataFrame, find_peaks: bool = True) -> List[int]:
        \"\"\"
        Find peaks or troughs in price data relative to the input DataFrame's index.
        
        Args:
            data: DataFrame with price data (e.g., a lookback slice)
            find_peaks: If True, find peaks (high points), otherwise find troughs (low points)
            
        Returns:
            List of relative indices (within the input data) of peaks or troughs
        \"\"\"
        # Use high for peaks and low for troughs
        price_series = data['high'] if find_peaks else data['low']
        
        # Minimum required deviation to consider a point as a peak/trough (as % of price)
        # Use a small absolute minimum deviation as well to handle low price assets
        min_deviation_pct = 0.005 * max(0.1, self.sensitivity) # Adjusted by sensitivity
        avg_price = price_series.mean()
        min_deviation_abs = avg_price * 0.001 # Small absolute minimum
        min_deviation = max(avg_price * min_deviation_pct, min_deviation_abs)
        
        # Find peaks or troughs using relative indices
        result_indices = []
        
        # Window size for local comparison (e.g., 2 bars on each side)
        window = 2
        
        # We need at least 2*window + 1 data points
        if len(data) < (2 * window + 1):
            return result_indices
            
        for i in range(window, len(data) - window):
            is_extremum = True
            for j in range(1, window + 1):
                if find_peaks:
                    # Check if current point is strictly higher than neighbors
                    if not (price_series.iloc[i] > price_series.iloc[i-j] and price_series.iloc[i] > price_series.iloc[i+j]):
                        is_extremum = False
                        break
                else:
                    # Check if current point is strictly lower than neighbors
                    if not (price_series.iloc[i] < price_series.iloc[i-j] and price_series.iloc[i] < price_series.iloc[i+j]):
                        is_extremum = False
                        break
            
            if is_extremum:
                # Check significance: difference from the max/min in the window
                window_slice = price_series.iloc[i-window : i+window+1]
                if find_peaks:
                    significance_check = price_series.iloc[i] - window_slice.drop(price_series.index[i]).max() > min_deviation
                else:
                    significance_check = window_slice.drop(price_series.index[i]).min() - price_series.iloc[i] > min_deviation

                if significance_check:
                    result_indices.append(i) # Append relative index
                    
        return result_indices
        
    def _calculate_trendline_loc(self, data: pd.DataFrame, locations: List[int], use_high: bool) -> Tuple[float, float]:
        \"\"\"
        Calculate a trendline through the selected points using absolute locations.
        
        Args:
            data: The full DataFrame with price data
            locations: List of absolute integer locations (iloc) to use for trendline calculation
            use_high: If True, use 'high' values, otherwise use 'low' values
            
        Returns:
            Tuple of (slope, intercept) where slope is per index unit (location)
        \"\"\"
        if not locations or len(locations) < 2:
            return 0, 0
            
        # Get x (locations) and y (prices) coordinates
        x_vals = np.array(locations)
        price_col = 'high' if use_high else 'low'
        # Ensure locations are valid before accessing .iloc
        valid_locations = [loc for loc in locations if 0 <= loc < len(data)]
        if len(valid_locations) < 2:
             return 0, 0
        y_vals = data[price_col].iloc[valid_locations].values
        x_vals = np.array(valid_locations) # Use only valid locations for fitting

        try:
            # Linear regression: y = slope * x + intercept
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
        except (np.linalg.LinAlgError, ValueError):
             # Handle cases where polyfit fails (e.g., all points collinear vertically)
             slope, intercept = 0, np.mean(y_vals) if len(y_vals) > 0 else 0

        return slope, intercept
        
    def _horizontal_line_fit_loc(self, data: pd.DataFrame, locations: List[int], is_resistance: bool) -> Optional[float]:
        \"\"\"
        Check if points align horizontally and calculate horizontal line using absolute locations.
        
        Args:
            data: The full DataFrame with price data
            locations: List of absolute integer locations (iloc) to check
            is_resistance: If True, check for horizontal resistance, otherwise support
            
        Returns:
            Horizontal line value if points align, None otherwise
        \"\"\"
        if not locations or len(locations) < 2:
            return None
            
        # Get price values using absolute locations
        price_col = 'high' if is_resistance else 'low'
        # Ensure locations are valid before accessing .iloc
        valid_locations = [loc for loc in locations if 0 <= loc < len(data)]
        if len(valid_locations) < 2:
             return None
        values = data[price_col].iloc[valid_locations]
        
        # Calculate statistics
        mean_val = values.mean()
        std_val = values.std()
        
        # Check if values are horizontal (small std dev relative to mean, adjusted by sensitivity)
        # Avoid division by zero
        if mean_val == 0: return None 
        is_horizontal = (std_val / mean_val) < (0.01 / max(self.sensitivity, 0.1))
        
        if is_horizontal:
            return mean_val
        else:
            return None
            
    def _find_contiguous_regions_loc(self, series: pd.Series) -> List[Tuple[int, int]]:
        \"\"\"
        Find contiguous regions where series values are 1, returning absolute locations.
        
        Args:
            series: Series with pattern markers (0 or 1) indexed like the main DataFrame
            
        Returns:
            List of (start_loc, end_loc) tuples for each contiguous region using iloc
        \"\"\"
        regions = []
        in_region = False
        start_loc = 0
        
        for i in range(len(series)):
            # Use .iloc for positional access
            if series.iloc[i] == 1 and not in_region:
                # Start of a new region
                in_region = True
                start_loc = i # Store the location
            elif series.iloc[i] != 1 and in_region:
                # End of a region
                in_region = False
                regions.append((start_loc, i - 1)) # End location is inclusive
                
        # Handle case where pattern extends to the end
        if in_region:
            regions.append((start_loc, len(series) - 1))
            
        return regions
        
    @property
    def projection_bars(self) -> int:
        \"\"\"Number of bars to project pattern components into the future.\"\""
        return 20  # Default value
        
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Chart Pattern Recognizer',
            'description': 'Identifies common chart patterns',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'lookback_period',
                    'description': 'Number of bars to look back for pattern recognition',
                    'type': 'int',
                    'default': 100
                },
                {
                    'name': 'pattern_types',
                    'description': 'List of pattern types to look for',
                    'type': 'list',
                    'default': None
                },
                {
                    'name': 'min_pattern_size',
                    'description': 'Minimum size of patterns to recognize (in bars)',
                    'type': 'int',
                    'default': 10
                },
                {
                    'name': 'max_pattern_size',
                    'description': 'Maximum size of patterns to recognize (in bars)',
                    'type': 'int',
                    'default': 50
                },
                {
                    'name': 'sensitivity',
                    'description': 'Sensitivity of pattern detection (0.0-1.0)',
                    'type': 'float',
                    'default': 0.75
                }
            ]
        }


class HarmonicPatternFinder(BaseIndicator):
    """
    Harmonic Pattern Finder
    
    Identifies harmonic price patterns like Gartley, Butterfly, Bat, Crab, etc.
    These patterns use Fibonacci ratios to identify potential reversal zones.
    """
    
    category = "pattern"
    
    def __init__(
        self, 
        lookback_period: int = 100,
        pattern_types: Optional[List[str]] = None,
        tolerance: float = 0.05,
        **kwargs
    ):
        """
        Initialize Harmonic Pattern Finder.
        
        Args:
            lookback_period: Number of bars to look back for pattern recognition
            pattern_types: List of pattern types to look for (None = all patterns)
            tolerance: Tolerance for Fibonacci ratio matches (0.01-0.10)
            **kwargs: Additional parameters
        """
        self.lookback_period = lookback_period
        self.tolerance = max(0.01, min(0.10, tolerance))
        
        # Set pattern types to recognize
        all_patterns = [
            "gartley", "butterfly", "bat", "crab", "shark", "cypher", "three_drives"
        ]
        
        if pattern_types is None:
            self.pattern_types = all_patterns
        else:
            self.pattern_types = [p for p in pattern_types if p in all_patterns]
            
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate harmonic pattern recognition for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with harmonic pattern values
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Initialize pattern columns with zeros
        for pattern in self.pattern_types:
            result[f"harmonic_{pattern}_bullish"] = 0
            result[f"harmonic_{pattern}_bearish"] = 0
            
        # Find significant pivot points
        window = max(2, self.lookback_period // 20)
        pivots = self._find_pivot_points(result, window)
        
        # We need at least 5 pivot points to form harmonic patterns
        if len(pivots) < 5:
            return result
            
        # Look for each requested pattern type
        # Assuming pivots is a list of (index, price) tuples
        for i in range(len(pivots) - 4):
            # Get 5 consecutive pivot points
            X, A, B, C, D = pivots[i:i+5]
            
            # Check for each pattern
            if "gartley" in self.pattern_types:
                self._find_gartley_pattern(result, X, A, B, C, D)
                
            if "butterfly" in self.pattern_types:
                self._find_butterfly_pattern(result, X, A, B, C, D)
                
            if "bat" in self.pattern_types:
                self._find_bat_pattern(result, X, A, B, C, D)
                
            if "crab" in self.pattern_types:
                self._find_crab_pattern(result, X, A, B, C, D)
                
            if "shark" in self.pattern_types:
                self._find_shark_pattern(result, X, A, B, C, D)
                
            if "cypher" in self.pattern_types:
                self._find_cypher_pattern(result, X, A, B, C, D)
                
            if "three_drives" in self.pattern_types:
                self._find_three_drives_pattern(result, X, A, B, C, D)
        
        return result
        
    def _find_pivot_points(self, data: pd.DataFrame, window: int) -> List[Tuple[int, float]]:
        """Find significant pivot points for harmonic pattern detection."""
        # This is a simplified implementation
        # A more sophisticated approach would involve a more robust pivot point detection algorithm
        
        # Find peaks and troughs in price data
        high_peaks, _ = self._find_peaks_and_troughs(data['high'], window)
        _, low_troughs = self._find_peaks_and_troughs(data['low'], window)
        
        # Combine peaks and troughs and sort by index
        pivots = []
        
        for i, (hp, lt) in enumerate(zip(high_peaks, low_troughs)):
            if hp == 1:
                pivots.append((i, data['high'].iloc[i]))
            elif lt == 1:
                pivots.append((i, data['low'].iloc[i]))
                
        # Sort by index
        pivots.sort(key=lambda x: x[0])
        
        return pivots
        
    def _find_peaks_and_troughs(self, prices: pd.Series, window: int = 5) -> Tuple[pd.Series, pd.Series]:
        """Find peaks and troughs in price data."""
        # Initialize peak and trough series
        peaks = pd.Series(0, index=prices.index)
        troughs = pd.Series(0, index=prices.index)
        
        # Find peaks and troughs
        for i in range(window, len(prices) - window):
            # Check if this is a peak
            if all(prices.iloc[i] > prices.iloc[i-j] for j in range(1, window+1)) and \
               all(prices.iloc[i] > prices.iloc[i+j] for j in range(1, window+1)):
                peaks.iloc[i] = 1
                
            # Check if this is a trough
            if all(prices.iloc[i] < prices.iloc[i-j] for j in range(1, window+1)) and \
               all(prices.iloc[i] < prices.iloc[i+j] for j in range(1, window+1)):
                troughs.iloc[i] = 1
                
        return peaks, troughs
        
    def _match_fib_ratio(self, actual: float, target: float) -> bool:
        """Check if a ratio matches a Fibonacci target within tolerance."""
        return abs(actual - target) <= self.tolerance
        
    def _find_gartley_pattern(self, result: pd.DataFrame, X, A, B, C, D):
        """Find bullish and bearish Gartley patterns."""
        # Implementation for Gartley pattern
        # Bullish Gartley: XA down, AB up, BC down, CD up
        # Bearish Gartley: XA up, AB down, BC up, CD down
        
        X_idx, X_price = X
        A_idx, A_price = A
        B_idx, B_price = B
        C_idx, C_price = C
        D_idx, D_price = D
        
        # Ensure correct sequence of pivots (X < A < B < C < D)
        if not (X_idx < A_idx < B_idx < C_idx < D_idx):
            return
            
        XA = abs(A_price - X_price)
        AB = abs(B_price - A_price)
        BC = abs(C_price - B_price)
        CD = abs(D_price - C_price)
        AD = abs(D_price - A_price)
        
        if XA == 0 or AB == 0 or BC == 0: return # Avoid division by zero

        # Ratios
        AB_XA = AB / XA
        BC_AB = BC / AB
        CD_BC = CD / BC
        AD_XA = AD / XA # Alternate D point check

        # Gartley Ratios
        gartley_AB_XA = 0.618
        gartley_BC_AB_min = 0.382
        gartley_BC_AB_max = 0.886
        gartley_CD_BC_min = 1.272
        gartley_CD_BC_max = 1.618
        gartley_AD_XA = 0.786 # D point should be 0.786 retracement of XA

        # Check Bullish Gartley (XA down, D is low pivot)
        is_bullish_structure = (X_price > A_price) and (B_price > A_price) and \
                               (C_price < B_price) and (D_price > C_price) and (D_price < B_price)
                               
        if is_bullish_structure:
            match_AB_XA = self._match_fib_ratio(AB_XA, gartley_AB_XA)
            match_BC_AB = gartley_BC_AB_min <= BC_AB <= gartley_BC_AB_max
            match_CD_BC = gartley_CD_BC_min <= CD_BC <= gartley_CD_BC_max
            match_AD_XA = self._match_fib_ratio(AD_XA, gartley_AD_XA)
            
            if match_AB_XA and match_BC_AB and match_CD_BC and match_AD_XA:
                 result.loc[D_idx, 'harmonic_gartley_bullish'] = 1

        # Check Bearish Gartley (XA up, D is high pivot)
        is_bearish_structure = (X_price < A_price) and (B_price < A_price) and \
                               (C_price > B_price) and (D_price < C_price) and (D_price > B_price)
                               
        if is_bearish_structure:
            match_AB_XA = self._match_fib_ratio(AB_XA, gartley_AB_XA)
            match_BC_AB = gartley_BC_AB_min <= BC_AB <= gartley_BC_AB_max
            match_CD_BC = gartley_CD_BC_min <= CD_BC <= gartley_CD_BC_max
            match_AD_XA = self._match_fib_ratio(AD_XA, gartley_AD_XA)
            
            if match_AB_XA and match_BC_AB and match_CD_BC and match_AD_XA:
                 result.loc[D_idx, 'harmonic_gartley_bearish'] = 1
        
    def _find_butterfly_pattern(self, result: pd.DataFrame, X, A, B, C, D):
        \"\"\"Find bullish and bearish Butterfly patterns.\"\"\"
        # Implementation for Butterfly pattern
        # Bullish Butterfly: XA down, AB up, BC down, CD up (D below X)
        # Bearish Butterfly: XA up, AB down, BC up, CD down (D above X)
        
        X_idx, X_price = X
        A_idx, A_price = A
        B_idx, B_price = B
        C_idx, C_price = C
        D_idx, D_price = D
        
        if not (X_idx < A_idx < B_idx < C_idx < D_idx): return
            
        XA = abs(A_price - X_price)
        AB = abs(B_price - A_price)
        BC = abs(C_price - B_price)
        CD = abs(D_price - C_price)
        AD = abs(D_price - A_price) # D relative to A
        XD = abs(D_price - X_price) # D relative to X

        if XA == 0 or AB == 0 or BC == 0: return

        # Ratios
        AB_XA = AB / XA
        BC_AB = BC / AB
        CD_BC = CD / BC
        XD_XA = XD / XA # D point is an extension of XA

        # Butterfly Ratios
        butterfly_AB_XA = 0.786
        butterfly_BC_AB_min = 0.382
        butterfly_BC_AB_max = 0.886
        butterfly_CD_BC_min = 1.618
        butterfly_CD_BC_max = 2.618
        butterfly_XD_XA_min = 1.272
        butterfly_XD_XA_max = 1.618

        # Check Bullish Butterfly (XA down, D is low pivot below X)
        is_bullish_structure = (X_price > A_price) and (B_price > A_price) and \
                               (C_price < B_price) and (D_price > C_price) and (D_price < A_price) # D below A
                               
        if is_bullish_structure and D_price < X_price: # D must be below X
            match_AB_XA = self._match_fib_ratio(AB_XA, butterfly_AB_XA)
            match_BC_AB = butterfly_BC_AB_min <= BC_AB <= butterfly_BC_AB_max
            match_CD_BC = butterfly_CD_BC_min <= CD_BC <= butterfly_CD_BC_max
            match_XD_XA = butterfly_XD_XA_min <= XD_XA <= butterfly_XD_XA_max
            
            if match_AB_XA and match_BC_AB and match_CD_BC and match_XD_XA:
                 result.loc[D_idx, 'harmonic_butterfly_bullish'] = 1

        # Check Bearish Butterfly (XA up, D is high pivot above X)
        is_bearish_structure = (X_price < A_price) and (B_price < A_price) and \
                               (C_price > B_price) and (D_price < C_price) and (D_price > A_price) # D above A
                               
        if is_bearish_structure and D_price > X_price: # D must be above X
            match_AB_XA = self._match_fib_ratio(AB_XA, butterfly_AB_XA)
            match_BC_AB = butterfly_BC_AB_min <= BC_AB <= butterfly_BC_AB_max
            match_CD_BC = butterfly_CD_BC_min <= CD_BC <= butterfly_CD_BC_max
            match_XD_XA = butterfly_XD_XA_min <= XD_XA <= butterfly_XD_XA_max
            
            if match_AB_XA and match_BC_AB and match_CD_BC and match_XD_XA:
                 result.loc[D_idx, 'harmonic_butterfly_bearish'] = 1
        
    def _find_bat_pattern(self, result: pd.DataFrame, X, A, B, C, D):
        \"\"\"Find bullish and bearish Bat patterns.\"\"\"
        # Implementation for Bat pattern
        # Bullish Bat: XA down, AB up, BC down, CD up (D near 0.886 XA)
        # Bearish Bat: XA up, AB down, BC up, CD down (D near 0.886 XA)
        
        X_idx, X_price = X
        A_idx, A_price = A
        B_idx, B_price = B
        C_idx, C_price = C
        D_idx, D_price = D
        
        if not (X_idx < A_idx < B_idx < C_idx < D_idx): return
            
        XA = abs(A_price - X_price)
        AB = abs(B_price - A_price)
        BC = abs(C_price - B_price)
        CD = abs(D_price - C_price)
        AD = abs(D_price - A_price) # D relative to A

        if XA == 0 or AB == 0 or BC == 0: return

        # Ratios
        AB_XA = AB / XA
        BC_AB = BC / AB
        CD_BC = CD / BC
        AD_XA = AD / XA # D point retracement of XA

        # Bat Ratios
        bat_AB_XA_min = 0.382
        bat_AB_XA_max = 0.5
        bat_BC_AB_min = 0.382
        bat_BC_AB_max = 0.886
        bat_CD_BC_min = 1.618
        bat_CD_BC_max = 2.618
        bat_AD_XA = 0.886

        # Check Bullish Bat (XA down, D is low pivot)
        is_bullish_structure = (X_price > A_price) and (B_price > A_price) and \
                               (C_price < B_price) and (D_price > C_price) and (D_price < A_price) # D below A
                               
        if is_bullish_structure:
            match_AB_XA = bat_AB_XA_min <= AB_XA <= bat_AB_XA_max
            match_BC_AB = bat_BC_AB_min <= BC_AB <= bat_BC_AB_max
            match_CD_BC = bat_CD_BC_min <= CD_BC <= bat_CD_BC_max
            match_AD_XA = self._match_fib_ratio(AD_XA, bat_AD_XA)
            
            if match_AB_XA and match_BC_AB and match_CD_BC and match_AD_XA:
                 result.loc[D_idx, 'harmonic_bat_bullish'] = 1

        # Check Bearish Bat (XA up, D is high pivot)
        is_bearish_structure = (X_price < A_price) and (B_price < A_price) and \
                               (C_price > B_price) and (D_price < C_price) and (D_price > A_price) # D above A
                               
        if is_bearish_structure:
            match_AB_XA = bat_AB_XA_min <= AB_XA <= bat_AB_XA_max
            match_BC_AB = bat_BC_AB_min <= BC_AB <= bat_BC_AB_max
            match_CD_BC = bat_CD_BC_min <= CD_BC <= bat_CD_BC_max
            match_AD_XA = self._match_fib_ratio(AD_XA, bat_AD_XA)
            
            if match_AB_XA and match_BC_AB and match_CD_BC and match_AD_XA:
                 result.loc[D_idx, 'harmonic_bat_bearish'] = 1
        
    def _find_crab_pattern(self, result: pd.DataFrame, X, A, B, C, D):
        \"\"\"Find bullish and bearish Crab patterns.\"\""
        # Implementation for Crab pattern (Deep Crab uses 0.886 AB/XA)
        # Bullish Crab: XA down, AB up, BC down, CD up (D deep extension 1.618 XA)
        # Bearish Crab: XA up, AB down, BC up, CD down (D deep extension 1.618 XA)
        
        X_idx, X_price = X
        A_idx, A_price = A
        B_idx, B_price = B
        C_idx, C_price = C
        D_idx, D_price = D
        
        if not (X_idx < A_idx < B_idx < C_idx < D_idx): return
            
        XA = abs(A_price - X_price)
        AB = abs(B_price - A_price)
        BC = abs(C_price - B_price)
        CD = abs(D_price - C_price)
        XD = abs(D_price - X_price) # D relative to X

        if XA == 0 or AB == 0 or BC == 0: return

        # Ratios
        AB_XA = AB / XA
        BC_AB = BC / AB
        CD_BC = CD / BC
        XD_XA = XD / XA # D point extension of XA

        # Crab Ratios
        crab_AB_XA_min = 0.382
        crab_AB_XA_max = 0.618
        crab_BC_AB_min = 0.382
        crab_BC_AB_max = 0.886
        crab_CD_BC_min = 2.24 # Can be up to 3.618
        crab_CD_BC_max = 3.618
        crab_XD_XA = 1.618

        # Check Bullish Crab (XA down, D is low pivot below X)
        is_bullish_structure = (X_price > A_price) and (B_price > A_price) and \
                               (C_price < B_price) and (D_price > C_price) and (D_price < X_price) # D below X
                               
        if is_bullish_structure:
            match_AB_XA = crab_AB_XA_min <= AB_XA <= crab_AB_XA_max
            match_BC_AB = crab_BC_AB_min <= BC_AB <= crab_BC_AB_max
            match_CD_BC = crab_CD_BC_min <= CD_BC <= crab_CD_BC_max
            match_XD_XA = self._match_fib_ratio(XD_XA, crab_XD_XA)
            
            if match_AB_XA and match_BC_AB and match_CD_BC and match_XD_XA:
                 result.loc[D_idx, 'harmonic_crab_bullish'] = 1

        # Check Bearish Crab (XA up, D is high pivot above X)
        is_bearish_structure = (X_price < A_price) and (B_price < A_price) and \
                               (C_price > B_price) and (D_price < C_price) and (D_price > X_price) # D above X
                               
        if is_bearish_structure:
            match_AB_XA = crab_AB_XA_min <= AB_XA <= crab_AB_XA_max
            match_BC_AB = crab_BC_AB_min <= BC_AB <= crab_BC_AB_max
            match_CD_BC = crab_CD_BC_min <= CD_BC <= crab_CD_BC_max
            match_XD_XA = self._match_fib_ratio(XD_XA, crab_XD_XA)
            
            if match_AB_XA and match_BC_AB and match_CD_BC and match_XD_XA:
                 result.loc[D_idx, 'harmonic_crab_bearish'] = 1
        
    def _find_shark_pattern(self, result: pd.DataFrame, X, A, B, C, D):
        \"\"\"Find bullish and bearish Shark patterns.\"\""
        # Note: Shark uses 0, X, A, B, C points. D is the completion point (C).
        # We adapt using X, A, B, C, D where D represents the C point of the Shark.
        # Bullish Shark: 0X up, XA down, AB up (extends beyond X), BC down (deep retracement/extension)
        # Bearish Shark: 0X down, XA up, AB down (extends beyond X), BC up (deep retracement/extension)
        
        # Let's rename pivots for clarity in Shark context: O, X, A, B, C
        O_idx, O_price = X # Using X as the '0' point
        X_idx, X_price = A # Using A as the 'X' point
        A_idx, A_price = B # Using B as the 'A' point
        B_idx, B_price = C # Using C as the 'B' point
        C_idx, C_price = D # Using D as the 'C' point (completion)

        if not (O_idx < X_idx < A_idx < B_idx < C_idx): return
            
        OX = abs(X_price - O_price)
        XA = abs(A_price - X_price)
        AB = abs(B_price - A_price)
        BC = abs(C_price - B_price)
        AC = abs(C_price - A_price) # C relative to A
        OC = abs(C_price - O_price) # C relative to O

        if OX == 0 or XA == 0 or AB == 0 or BC == 0: return

        # Ratios
        AB_XA = AB / XA
        BC_AB = BC / AB
        OC_OB = OC / abs(B_price - O_price)# C relative to OB leg (0B)

        # Shark Ratios
        shark_AB_XA_min = 1.13
        shark_AB_XA_max = 1.618
        shark_BC_AB_min = 1.618
        shark_BC_AB_max = 2.24
        shark_OC_OB_min = 0.886 # C retracement of 0B
        shark_OC_OB_max = 1.13  # C extension of 0B

        # Check Bullish Shark (0X up, C is low pivot)
        is_bullish_structure = (X_price > O_price) and (A_price < X_price) and \
                               (B_price > X_price) and (C_price < A_price) # B extends beyond X, C below A
                               
        if is_bullish_structure:
            match_AB_XA = shark_AB_XA_min <= AB_XA <= shark_AB_XA_max
            match_BC_AB = shark_BC_AB_min <= BC_AB <= shark_BC_AB_max
            match_OC_OB = shark_OC_OB_min <= OC_OB <= shark_OC_OB_max
            
            if match_AB_XA and match_BC_AB and match_OC_OB:
                 result.loc[C_idx, 'harmonic_shark_bullish'] = 1 # Mark at C point

        # Check Bearish Shark (0X down, C is high pivot)
        is_bearish_structure = (X_price < O_price) and (A_price > X_price) and \
                               (B_price < X_price) and (C_price > A_price) # B extends beyond X, C above A
                               
        if is_bearish_structure:
            match_AB_XA = shark_AB_XA_min <= AB_XA <= shark_AB_XA_max
            match_BC_AB = shark_BC_AB_min <= BC_AB <= shark_BC_AB_max
            match_OC_OB = shark_OC_OB_min <= OC_OB <= shark_OC_OB_max
            
            if match_AB_XA and match_BC_AB and match_OC_OB:
                 result.loc[C_idx, 'harmonic_shark_bearish'] = 1 # Mark at C point
        
    def _find_cypher_pattern(self, result: pd.DataFrame, X, A, B, C, D):
        \"\"\"Find bullish and bearish Cypher patterns.\"\""
        # Bullish Cypher: XA up, AB down (retraces XA), BC up (extends beyond A), CD down (retraces XC)
        # Bearish Cypher: XA down, AB up (retraces XA), BC down (extends beyond A), CD up (retraces XC)
        
        X_idx, X_price = X
        A_idx, A_price = A
        B_idx, B_price = B
        C_idx, C_price = C
        D_idx, D_price = D
        
        if not (X_idx < A_idx < B_idx < C_idx < D_idx): return
            
        XA = abs(A_price - X_price)
        AB = abs(B_price - A_price) # Retracement leg
        XC = abs(C_price - X_price) # Used for D point calculation
        BC = abs(C_price - B_price) # Extension leg
        CD = abs(D_price - C_price) # Final retracement leg

        if XA == 0 or XC == 0 or BC == 0: return

        # Ratios
        AB_XA = AB / XA
        BC_AB = BC / AB # Not standard for Cypher, C is extension of XA
        C_ext_XA = abs(C_price - A_price) / XA # C point extension relative to XA
        CD_XC = CD / XC # D point retracement of XC leg

        # Cypher Ratios
        cypher_AB_XA_min = 0.382
        cypher_AB_XA_max = 0.618
        cypher_C_ext_XA_min = 1.272 # C must extend beyond A relative to XA
        cypher_C_ext_XA_max = 1.414
        cypher_CD_XC = 0.786 # D is 0.786 retracement of XC

        # Check Bullish Cypher (XA up, D is low pivot between X and C)
        is_bullish_structure = (A_price > X_price) and (B_price < A_price) and (B_price > X_price) and \
                               (C_price > A_price) and (D_price < C_price) and (D_price > X_price) # C extends A, D retraces XC
                               
        if is_bullish_structure:
            match_AB_XA = cypher_AB_XA_min <= AB_XA <= cypher_AB_XA_max
            match_C_ext_XA = cypher_C_ext_XA_min <= C_ext_XA <= cypher_C_ext_XA_max
            match_CD_XC = self._match_fib_ratio(CD_XC, cypher_CD_XC)
            
            if match_AB_XA and match_C_ext_XA and match_CD_XC:
                 result.loc[D_idx, 'harmonic_cypher_bullish'] = 1

        # Check Bearish Cypher (XA down, D is high pivot between X and C)
        is_bearish_structure = (A_price < X_price) and (B_price > A_price) and (B_price < X_price) and \
                               (C_price < A_price) and (D_price > C_price) and (D_price < X_price) # C extends A, D retraces XC
                               
        if is_bearish_structure:
            match_AB_XA = cypher_AB_XA_min <= AB_XA <= cypher_AB_XA_max
            match_C_ext_XA = cypher_C_ext_XA_min <= C_ext_XA <= cypher_C_ext_XA_max
            match_CD_XC = self._match_fib_ratio(CD_XC, cypher_CD_XC)
            
            if match_AB_XA and match_C_ext_XA and match_CD_XC:
                 result.loc[D_idx, 'harmonic_cypher_bearish'] = 1
        
    def _find_three_drives_pattern(self, result: pd.DataFrame, X, A, B, C, D):
        \"\"\"Find bullish and bearish Three Drives patterns.\"\""
        # Requires 6 points (0, 1, A, 2, B, 3). We use 5 pivots X,A,B,C,D to approximate Drive 1, A, Drive 2, B, Drive 3
        # Drive 1 = XA, Correction A = AB, Drive 2 = BC, Correction B = CD, Drive 3 = DE (needs one more pivot)
        # Simplified: Look for symmetry in drives and corrections using 5 points.
        # Drive 1 (XA), Correction 1 (AB), Drive 2 (BC), Correction 2 (CD), Start of Drive 3 (D)
        
        # This pattern is harder to implement reliably with only 5 pivots.
        # A basic check for symmetry:
        # Drive 2 approx 1.272 or 1.618 of Correction 1 (BC / AB)
        # Drive 3 approx 1.272 or 1.618 of Correction 2 (Requires point E)
        # Correction A approx 0.618 or 0.786 of Drive 1 (AB / XA)
        # Correction B approx 0.618 or 0.786 of Drive 2 (CD / BC)
        
        X_idx, X_price = X
        A_idx, A_price = A
        B_idx, B_price = B
        C_idx, C_price = C
        D_idx, D_price = D
        
        if not (X_idx < A_idx < B_idx < C_idx < D_idx): return
            
        Drive1 = abs(A_price - X_price)
        CorrA = abs(B_price - A_price)
        Drive2 = abs(C_price - B_price)
        CorrB = abs(D_price - C_price)

        if Drive1 == 0 or CorrA == 0 or Drive2 == 0 or CorrB == 0: return

        # Ratios
        CorrA_Drive1 = CorrA / Drive1
        CorrB_Drive2 = CorrB / Drive2
        Drive2_CorrA = Drive2 / CorrA

        # Three Drives Ratios (approximate)
        drive_retracement_min = 0.618
        drive_retracement_max = 0.786
        drive_extension_min = 1.272
        drive_extension_max = 1.618

        # Check Bullish Three Drives (Overall upward trend)
        is_bullish_structure = (A_price > X_price) and (B_price < A_price) and \
                               (C_price > B_price) and (D_price < C_price) and (C_price > A_price) # Higher highs and lows
                               
        if is_bullish_structure:
            match_CorrA = self._match_fib_ratio(CorrA_Drive1, drive_retracement_min) or \
                          self._match_fib_ratio(CorrA_Drive1, drive_retracement_max)
            match_CorrB = self._match_fib_ratio(CorrB_Drive2, drive_retracement_min) or \
                          self._match_fib_ratio(CorrB_Drive2, drive_retracement_max)
            match_Drive2 = self._match_fib_ratio(Drive2_CorrA, drive_extension_min) or \
                           self._match_fib_ratio(Drive2_CorrA, drive_extension_max)
            
            # If first two drives and corrections match, mark potential start of Drive 3 at D
            if match_CorrA and match_CorrB and match_Drive2:
                 result.loc[D_idx, 'harmonic_three_drives_bullish'] = 1 # Mark at D (start of potential Drive 3)

        # Check Bearish Three Drives (Overall downward trend)
        is_bearish_structure = (A_price < X_price) and (B_price > A_price) and \
                               (C_price < B_price) and (D_price > C_price) and (C_price < A_price) # Lower lows and highs
                               
        if is_bearish_structure:
            match_CorrA = self._match_fib_ratio(CorrA_Drive1, drive_retracement_min) or \
                          self._match_fib_ratio(CorrA_Drive1, drive_retracement_max)
            match_CorrB = self._match_fib_ratio(CorrB_Drive2, drive_retracement_min) or \
                          self._match_fib_ratio(CorrB_Drive2, drive_retracement_max)
            match_Drive2 = self._match_fib_ratio(Drive2_CorrA, drive_extension_min) or \
                           self._match_fib_ratio(Drive2_CorrA, drive_extension_max)

            if match_CorrA and match_CorrB and match_Drive2:
                 result.loc[D_idx, 'harmonic_three_drives_bearish'] = 1 # Mark at D (start of potential Drive 3)
        
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Harmonic Pattern Finder',
            'description': 'Identifies harmonic price patterns based on Fibonacci ratios',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'lookback_period',
                    'description': 'Number of bars to look back for pattern recognition',
                    'type': 'int',
                    'default': 100
                },
                {
                    'name': 'pattern_types',
                    'description': 'List of pattern types to look for (None = all patterns)',
                    'type': 'list',
                    'default': None
                },
                {
                    'name': 'tolerance',
                    'description': 'Tolerance for Fibonacci ratio matches (0.01-0.10)',
                    'type': 'float',
                    'default': 0.05
                }
            ]
        }


class CandlestickPatterns(BaseIndicator):
    """
    Candlestick Pattern Recognition
    
    Identifies common candlestick patterns that may indicate trend continuations
    or reversals.
    """
    
    category = "pattern"
    
    def __init__(
        self, 
        pattern_types: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize Candlestick Pattern Recognition.
        
        Args:
            pattern_types: List of pattern types to look for (None = all patterns)
            **kwargs: Additional parameters
        """
        # Set pattern types to recognize
        all_patterns = [
            "doji", "hammer", "hanging_man", "engulfing", "morning_star", "evening_star",
            "three_white_soldiers", "three_black_crows", "spinning_top", 
            "harami", "piercing_line", "dark_cloud_cover"
        ]
        
        if pattern_types is None:
            self.pattern_types = all_patterns
        else:
            self.pattern_types = [p for p in pattern_types if p in all_patterns]
            
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate candlestick pattern recognition for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with candlestick pattern values
        """
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Initialize pattern columns with zeros
        for pattern in self.pattern_types:
            if pattern == "doji":
                result[f"candle_doji"] = 0
            else:
                result[f"candle_{pattern}_bullish"] = 0
                result[f"candle_{pattern}_bearish"] = 0
                
        # Calculate necessary values
        result['body_size'] = abs(result['close'] - result['open'])
        result['range'] = result['high'] - result['low']
        result['body_percentage'] = result['body_size'] / result['range']
        result['upper_shadow'] = result.apply(
            lambda x: x['high'] - max(x['open'], x['close']), axis=1
        )
        result['lower_shadow'] = result.apply(
            lambda x: min(x['open'], x['close']) - x['low'], axis=1
        )
        
        # Identify candlestick patterns
        if "doji" in self.pattern_types:
            result = self._identify_doji(result)
            
        if "hammer" in self.pattern_types:
            result = self._identify_hammer(result)
            
        if "hanging_man" in self.pattern_types:
            result = self._identify_hanging_man(result)
            
        if "engulfing" in self.pattern_types:
            result = self._identify_engulfing(result)
            
        if "morning_star" in self.pattern_types:
            result = self._identify_morning_star(result)
            
        if "evening_star" in self.pattern_types:
            result = self._identify_evening_star(result)
            
        if "three_white_soldiers" in self.pattern_types:
            result = self._identify_three_white_soldiers(result)
            
        if "three_black_crows" in self.pattern_types:
            result = self._identify_three_black_crows(result)
            
        if "spinning_top" in self.pattern_types:
            result = self._identify_spinning_top(result)
            
        if "harami" in self.pattern_types:
            result = self._identify_harami(result)
            
        if "piercing_line" in self.pattern_types:
            result = self._identify_piercing_line(result)
            
        if "dark_cloud_cover" in self.pattern_types:
            result = self._identify_dark_cloud_cover(result)
            
        # Drop intermediate columns
        result = result.drop(['body_size', 'range', 'body_percentage', 
                             'upper_shadow', 'lower_shadow'], axis=1)
            
        return result
        
    def _identify_doji(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identify doji candlestick pattern."""
        # Doji has very small body compared to range
        data.loc[data['body_percentage'] < 0.1, 'candle_doji'] = 1
        return data
        
    def _identify_hammer(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identify hammer pattern."""
        # Hammer has small body, little/no upper shadow, and long lower shadow in a downtrend
        df = data.copy()
        
        # Check for hammer conditions
        body_small = df['body_percentage'] < 0.3
        lower_shadow_long = df['lower_shadow'] > 2 * df['body_size']
        upper_shadow_small = df['upper_shadow'] < 0.1 * df['range']
        downtrend = df['close'].shift(1) < df['close'].shift(5)
        
        # Set bullish hammer
        bullish_hammer = body_small & lower_shadow_long & upper_shadow_small & downtrend
        df.loc[bullish_hammer, 'candle_hammer_bullish'] = 1
        
        return df
        
    def _identify_hanging_man(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identify hanging man pattern."""
        # Hanging man similar to hammer but in an uptrend and bearish
        df = data.copy()
        
        # Check for hanging man conditions
        body_small = df['body_percentage'] < 0.3
        lower_shadow_long = df['lower_shadow'] > 2 * df['body_size']
        upper_shadow_small = df['upper_shadow'] < 0.1 * df['range']
        uptrend = df['close'].shift(1) > df['close'].shift(5)
        
        # Set bearish hanging man
        bearish_hanging_man = body_small & lower_shadow_long & upper_shadow_small & uptrend
        df.loc[bearish_hanging_man, 'candle_hanging_man_bearish'] = 1
        
        return df
        
    def _identify_engulfing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identify bullish and bearish engulfing patterns."""
        df = data.copy()
        
        # Bullish engulfing
        bullish_engulfing = (
            (df['close'].shift(1) < df['open'].shift(1)) &  # Previous candle is bearish
            (df['close'] > df['open']) &  # Current candle is bullish
            (df['open'] <= df['close'].shift(1)) &  # Current open is lower than or equal to prev close
            (df['close'] >= df['open'].shift(1))  # Current close is higher than or equal to prev open
        )
        df.loc[bullish_engulfing, 'candle_engulfing_bullish'] = 1
        
        # Bearish engulfing
        bearish_engulfing = (
            (df['close'].shift(1) > df['open'].shift(1)) &  # Previous candle is bullish
            (df['close'] < df['open']) &  # Current candle is bearish
            (df['open'] >= df['close'].shift(1)) &  # Current open is higher than or equal to prev close
            (df['close'] <= df['open'].shift(1))  # Current close is lower than or equal to prev open
        )
        df.loc[bearish_engulfing, 'candle_engulfing_bearish'] = 1
        
        return df
        
    def _identify_morning_star(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Identify morning star pattern.\"\"\"
        # Placeholder implementation - this would require examining 3 candles
        df = data.copy()
        
        # Conditions for Morning Star (Bullish Reversal)
        # 1. Previous trend is downtrend (optional check, often assumed)
        # 2. Candle 1: Long bearish candle
        # 3. Candle 2: Small body (bullish or bearish), gaps down from Candle 1
        # 4. Candle 3: Long bullish candle, closes well into Candle 1's body (above midpoint)
        
        c1_bearish = df['close'].shift(2) < df['open'].shift(2)
        c1_long_body = df['body_size'].shift(2) > df['body_size'].rolling(10).mean().shift(2) # Compare to recent average
        
        c2_small_body = df['body_size'].shift(1) < df['body_size'].rolling(10).mean().shift(1) * 0.5
        c2_gap_down = df['high'].shift(1) < df['low'].shift(2) # Top of C2 below bottom of C1
        
        c3_bullish = df['close'] > df['open']
        c3_long_body = df['body_size'] > df['body_size'].rolling(10).mean() * 0.5
        c3_closes_in_c1 = df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2 # Closes above midpoint of C1 body
        
        morning_star = c1_bearish & c1_long_body & c2_small_body & c2_gap_down & c3_bullish & c3_long_body & c3_closes_in_c1
        
        df.loc[morning_star, 'candle_morning_star_bullish'] = 1
        
        return df
        
    def _identify_evening_star(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Identify evening star pattern.\"\""
        # Placeholder implementation - this would require examining 3 candles
        df = data.copy()
        
        # Conditions for Evening Star (Bearish Reversal)
        # 1. Previous trend is uptrend (optional check, often assumed)
        # 2. Candle 1: Long bullish candle
        # 3. Candle 2: Small body (bullish or bearish), gaps up from Candle 1
        # 4. Candle 3: Long bearish candle, closes well into Candle 1's body (below midpoint)
        
        c1_bullish = df['close'].shift(2) > df['open'].shift(2)
        c1_long_body = df['body_size'].shift(2) > df['body_size'].rolling(10).mean().shift(2)
        
        c2_small_body = df['body_size'].shift(1) < df['body_size'].rolling(10).mean().shift(1) * 0.5
        c2_gap_up = df['low'].shift(1) > df['high'].shift(2) # Bottom of C2 above top of C1
        
        c3_bearish = df['close'] < df['open']
        c3_long_body = df['body_size'] > df['body_size'].rolling(10).mean() * 0.5
        c3_closes_in_c1 = df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2 # Closes below midpoint of C1 body
        
        evening_star = c1_bullish & c1_long_body & c2_small_body & c2_gap_up & c3_bearish & c3_long_body & c3_closes_in_c1
        
        df.loc[evening_star, 'candle_evening_star_bearish'] = 1
        
        return df
        
    def _identify_three_white_soldiers(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Identify three white soldiers pattern.\"\""
        # Placeholder implementation - this would require examining 3 candles
        df = data.copy()
        
        # Conditions for Three White Soldiers (Bullish Continuation/Reversal)
        # 1. Three consecutive long bullish candles
        # 2. Each candle closes higher than the previous candle
        # 3. Each candle opens within the body of the previous candle
        # 4. Candles should have small upper shadows (ideally)
        
        c1_bullish = df['close'].shift(2) > df['open'].shift(2)
        c2_bullish = df['close'].shift(1) > df['open'].shift(1)
        c3_bullish = df['close'] > df['open']
        
        c1_long = df['body_size'].shift(2) > df['body_size'].rolling(10).mean().shift(2) * 0.7
        c2_long = df['body_size'].shift(1) > df['body_size'].rolling(10).mean().shift(1) * 0.7
        c3_long = df['body_size'] > df['body_size'].rolling(10).mean() * 0.7
        
        higher_closes = (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2))
        
        opens_in_body = (df['open'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1)) & \
                        (df['open'].shift(1) > df['open'].shift(2)) & (df['open'].shift(1) < df['close'].shift(2))
                        
        # Optional: Check for small upper shadows
        # c1_small_upper = df['upper_shadow'].shift(2) < df['body_size'].shift(2) * 0.3
        # c2_small_upper = df['upper_shadow'].shift(1) < df['body_size'].shift(1) * 0.3
        # c3_small_upper = df['upper_shadow'] < df['body_size'] * 0.3
        
        three_white_soldiers = c1_bullish & c2_bullish & c3_bullish & \
                               c1_long & c2_long & c3_long & \
                               higher_closes & opens_in_body # & c1_small_upper & c2_small_upper & c3_small_upper
                               
        df.loc[three_white_soldiers, 'candle_three_white_soldiers_bullish'] = 1
        
        return df
        
    def _identify_three_black_crows(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Identify three black crows pattern.\"\""
        # Placeholder implementation - this would require examining 3 candles
        df = data.copy()
        
        # Conditions for Three Black Crows (Bearish Continuation/Reversal)
        # 1. Three consecutive long bearish candles
        # 2. Each candle closes lower than the previous candle
        # 3. Each candle opens within the body of the previous candle
        # 4. Candles should have small lower shadows (ideally)
        
        c1_bearish = df['close'].shift(2) <df['open'].shift(2)
        c2_bearish = df['close'].shift(1) < df['open'].shift(1)
        c3_bearish = df['close'] < df['open']
        
        c1_long = df['body_size'].shift(2) > df['body_size'].rolling(10).mean().shift(2) * 0.7
        c2_long = df['body_size'].shift(1) > df['body_size'].rolling(10).mean().shift(1) * 0.7
        c3_long = df['body_size'] > df['body_size'].rolling(10).mean() * 0.7
        
        lower_closes = (df['close'] < df['close'].shift(1)) & (df['close'].shift(1) < df['close'].shift(2))
        
        opens_in_body = (df['open'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1)) & \
                        (df['open'].shift(1) < df['open'].shift(2)) & (df['open'].shift(1) > df['close'].shift(2))
                        
        # Optional: Check for small lower shadows
        # c1_small_lower = df['lower_shadow'].shift(2) < df['body_size'].shift(2) * 0.3
        # c2_small_lower = df['lower_shadow'].shift(1) < df['body_size'].shift(1) * 0.3
        # c3_small_lower = df['lower_shadow'] < df['body_size'] * 0.3
        
        three_black_crows = c1_bearish & c2_bearish & c3_bearish & \
                            c1_long & c2_long & c3_long & \
                            lower_closes & opens_in_body # & c1_small_lower & c2_small_lower & c3_small_lower
                            
        df.loc[three_black_crows, 'candle_three_black_crows_bearish'] = 1
        
        return df
        
    def _identify_spinning_top(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Identify spinning top pattern.\"\""
        df = data.copy()
        
        # Spinning top has small body with upper and lower shadows roughly similar in size
        body_small = df['body_percentage'] < 0.3
        # Ensure shadows are significant compared to the body
        upper_shadow_significant = df['upper_shadow'] > df['body_size'] * 0.5 
        lower_shadow_significant = df['lower_shadow'] > df['body_size'] * 0.5
        # Ensure shadows are not excessively long compared to the range
        upper_shadow_not_extreme = df['upper_shadow'] < df['range'] * 0.7
        lower_shadow_not_extreme = df['lower_shadow'] < df['range'] * 0.7

        spinning_top_condition = body_small & upper_shadow_significant & lower_shadow_significant & upper_shadow_not_extreme & lower_shadow_not_extreme
        
        # Set bullish (close > open) and bearish (close < open) spinning tops
        bullish_spinning_top = spinning_top_condition & (df['close'] > df['open'])
        bearish_spinning_top = spinning_top_condition & (df['close'] < df['open'])
        
        df.loc[bullish_spinning_top, 'candle_spinning_top_bullish'] = 1
        df.loc[bearish_spinning_top, 'candle_spinning_top_bearish'] = 1
        
        return df
        
    def _identify_harami(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Identify harami pattern.\"\""
        # Placeholder implementation
        df = data.copy()
        
        # Conditions for Harami (Reversal)
        # 1. Candle 1: Large body
        # 2. Candle 2: Small body, completely contained within Candle 1's body
        
        c1_large_body = df['body_size'].shift(1) > df['body_size'].rolling(10).mean().shift(1)
        c2_small_body = df['body_size'] < df['body_size'].rolling(10).mean() * 0.5
        
        # Body containment check
        c2_inside_c1_body = (
            (df['open'] < df['open'].shift(1)) & (df['close'] < df['open'].shift(1)) & # C2 open below C1 open
            (df['open'] > df['close'].shift(1)) & (df['close'] > df['close'].shift(1))   # C2 open above C1 close
        ) | ( # Handles both bullish and bearish C1
            (df['open'] < df['close'].shift(1)) & (df['close'] < df['close'].shift(1)) & # C2 open below C1 close
            (df['open'] > df['open'].shift(1)) & (df['close'] > df['open'].shift(1))     # C2 open above C1 open
        )

        # Bullish Harami: C1 is bearish, C2 is bullish
        bullish_harami = c1_large_body & c2_small_body & c2_inside_c1_body & \
                         (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open'])
                         
        # Bearish Harami: C1 is bullish, C2 is bearish
        bearish_harami = c1_large_body & c2_small_body & c2_inside_c1_body & \
                         (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open'])
                         
        df.loc[bullish_harami, 'candle_harami_bullish'] = 1
        df.loc[bearish_harami, 'candle_harami_bearish'] = 1
        
        return df
        
    def _identify_piercing_line(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Identify piercing line pattern.\"\""
        # Placeholder implementation
        df = data.copy()
        
        # Conditions for Piercing Line (Bullish Reversal)
        # 1. Candle 1: Bearish candle
        # 2. Candle 2: Bullish candle
        # 3. Candle 2 opens below Candle 1's low
        # 4. Candle 2 closes above the midpoint of Candle 1's body
        
        c1_bearish = df['close'].shift(1) < df['open'].shift(1)
        c2_bullish = df['close'] > df['open']
        
        c2_opens_below_c1_low = df['open'] < df['low'].shift(1)
        c2_closes_above_midpoint = df['close'] > (df['open'].shift(1) + df['close'].shift(1)) / 2
        # Ensure C2 doesn't close above C1 open (would be engulfing)
        c2_not_engulfing = df['close'] < df['open'].shift(1) 

        piercing_line = c1_bearish & c2_bullish & c2_opens_below_c1_low & c2_closes_above_midpoint & c2_not_engulfing
        
        df.loc[piercing_line, 'candle_piercing_line_bullish'] = 1
        
        return df
        
    def _identify_dark_cloud_cover(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Identify dark cloud cover pattern.\"\""
        # Placeholder implementation
        df = data.copy()
        
        # Conditions for Dark Cloud Cover (Bearish Reversal)
        # 1. Candle 1: Bullish candle
        # 2. Candle 2: Bearish candle
        # 3. Candle 2 opens above Candle 1's high
        # 4. Candle 2 closes below the midpoint of Candle 1's body
        
        c1_bullish = df['close'].shift(1) > df['open'].shift(1)
        c2_bearish = df['close'] < df['open']
        
        c2_opens_above_c1_high = df['open'] > df['high'].shift(1)
        c2_closes_below_midpoint = df['close'] < (df['open'].shift(1) + df['close'].shift(1)) / 2
        # Ensure C2 doesn't close below C1 open (would be engulfing)
        c2_not_engulfing = df['close'] > df['open'].shift(1)

        dark_cloud_cover = c1_bullish & c2_bearish & c2_opens_above_c1_high & c2_closes_below_midpoint & c2_not_engulfing
        
        df.loc[dark_cloud_cover, 'candle_dark_cloud_cover_bearish'] = 1
        
        return df
        
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Candlestick Pattern Recognition',
            'description': 'Identifies common candlestick patterns for technical analysis',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'pattern_types',
                    'description': 'List of pattern types to look for (None = all patterns)',
                    'type': 'list',
                    'default': None
                }
            ]
        }
