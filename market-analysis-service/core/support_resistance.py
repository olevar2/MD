"""
Support and Resistance module for Market Analysis Service.

This module provides algorithms for identifying support and resistance levels in market data.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
from market_analysis_service.models.market_analysis_models import SupportResistanceMethod

logger = logging.getLogger(__name__)

class SupportResistanceDetector:
    """
    Class for detecting support and resistance levels in market data.
    """
    
    def __init__(self):
        """
        Initialize the Support and Resistance Detector.
        """
        self.available_methods = self._get_available_methods()
        
    def _get_available_methods(self) -> List[Dict[str, Any]]:
        """
        Get available methods for support and resistance detection.
        
        Returns:
            List of available methods
        """
        methods = []
        
        for method_type in SupportResistanceMethod:
            method_info = {
                "id": method_type.value,
                "name": method_type.name,
                "description": self._get_method_description(method_type)
            }
            
            methods.append(method_info)
            
        return methods
        
    def _get_method_description(self, method_type: SupportResistanceMethod) -> str:
        """
        Get description for a method type.
        
        Args:
            method_type: Method type
            
        Returns:
            Method description
        """
        descriptions = {
            SupportResistanceMethod.PRICE_SWINGS: "Detect support and resistance levels based on price swings",
            SupportResistanceMethod.MOVING_AVERAGE: "Use moving averages to identify support and resistance levels",
            SupportResistanceMethod.FIBONACCI: "Use Fibonacci retracement levels as support and resistance",
            SupportResistanceMethod.PIVOT_POINTS: "Calculate pivot points for support and resistance levels",
            SupportResistanceMethod.VOLUME_PROFILE: "Use volume profile to identify support and resistance levels",
            SupportResistanceMethod.FRACTAL: "Use fractals to identify support and resistance levels",
            SupportResistanceMethod.CUSTOM: "Custom method defined by user parameters"
        }
        
        return descriptions.get(method_type, "Unknown method")
        
    def identify_support_resistance(
        self,
        data: pd.DataFrame,
        methods: List[SupportResistanceMethod],
        levels_count: int = 5,
        additional_parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify support and resistance levels in market data.
        
        Args:
            data: Market data
            methods: Methods to use for identification
            levels_count: Number of levels to identify
            additional_parameters: Additional parameters for identification
            
        Returns:
            List of identified levels
        """
        if additional_parameters is None:
            additional_parameters = {}
            
        all_levels = []
        
        for method in methods:
            # Skip custom method if no parameters are provided
            if method == SupportResistanceMethod.CUSTOM and "custom_method_params" not in additional_parameters:
                continue
                
            # Identify levels using the method
            levels = self._identify_levels_with_method(data, method, additional_parameters)
            
            all_levels.extend(levels)
            
        # Sort levels by strength (descending)
        all_levels.sort(key=lambda x: x["strength"], reverse=True)
        
        # Remove duplicate levels (within a small range)
        unique_levels = self._remove_duplicate_levels(all_levels)
        
        # Limit the number of levels
        return unique_levels[:levels_count]
        
    def _identify_levels_with_method(
        self,
        data: pd.DataFrame,
        method: SupportResistanceMethod,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify support and resistance levels using a specific method.
        
        Args:
            data: Market data
            method: Method to use
            parameters: Additional parameters
            
        Returns:
            List of identified levels
        """
        levels = []
        
        try:
            if method == SupportResistanceMethod.PRICE_SWINGS:
                levels = self._identify_levels_price_swings(data, parameters)
                
            elif method == SupportResistanceMethod.MOVING_AVERAGE:
                levels = self._identify_levels_moving_average(data, parameters)
                
            elif method == SupportResistanceMethod.FIBONACCI:
                levels = self._identify_levels_fibonacci(data, parameters)
                
            elif method == SupportResistanceMethod.PIVOT_POINTS:
                levels = self._identify_levels_pivot_points(data, parameters)
                
            elif method == SupportResistanceMethod.VOLUME_PROFILE:
                levels = self._identify_levels_volume_profile(data, parameters)
                
            elif method == SupportResistanceMethod.FRACTAL:
                levels = self._identify_levels_fractal(data, parameters)
                
            elif method == SupportResistanceMethod.CUSTOM:
                if "custom_method_params" in parameters:
                    levels = self._identify_levels_custom(data, parameters["custom_method_params"])
                    
        except Exception as e:
            logger.error(f"Error identifying levels with {method.value} method: {e}")
            
        return levels
        
    def _identify_levels_price_swings(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify support and resistance levels using price swings method.
        
        Args:
            data: Market data
            parameters: Additional parameters
            
        Returns:
            List of identified levels
        """
        levels = []
        
        # Get parameters
        window = parameters.get("window", 5)
        min_touches = parameters.get("min_touches", 2)
        
        # Find local maxima and minima
        highs = data["high"].values
        lows = data["low"].values
        
        local_maxima = self._find_local_maxima(highs, window)
        local_minima = self._find_local_minima(lows, window)
        
        # Group similar levels
        resistance_levels = self._group_similar_levels(highs, local_maxima, window)
        support_levels = self._group_similar_levels(lows, local_minima, window)
        
        # Create resistance levels
        for level, touches in resistance_levels.items():
            if touches >= min_touches:
                # Find the last touch date
                last_touch_idx = max([i for i in local_maxima if abs(highs[i] - level) / level < 0.01])
                last_touch_date = data.index[last_touch_idx].isoformat() if hasattr(data.index[last_touch_idx], 'isoformat') else str(data.index[last_touch_idx])
                
                levels.append({
                    "price": float(level),
                    "type": "resistance",
                    "strength": float(touches / len(data) * 100),  # Normalize strength
                    "method": SupportResistanceMethod.PRICE_SWINGS.value,
                    "touches": int(touches),
                    "last_touch_date": last_touch_date
                })
                
        # Create support levels
        for level, touches in support_levels.items():
            if touches >= min_touches:
                # Find the last touch date
                last_touch_idx = max([i for i in local_minima if abs(lows[i] - level) / level < 0.01])
                last_touch_date = data.index[last_touch_idx].isoformat() if hasattr(data.index[last_touch_idx], 'isoformat') else str(data.index[last_touch_idx])
                
                levels.append({
                    "price": float(level),
                    "type": "support",
                    "strength": float(touches / len(data) * 100),  # Normalize strength
                    "method": SupportResistanceMethod.PRICE_SWINGS.value,
                    "touches": int(touches),
                    "last_touch_date": last_touch_date
                })
                
        return levels
        
    def _identify_levels_moving_average(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify support and resistance levels using moving average method.
        
        Args:
            data: Market data
            parameters: Additional parameters
            
        Returns:
            List of identified levels
        """
        levels = []
        
        # Get parameters
        ma_periods = parameters.get("ma_periods", [20, 50, 100, 200])
        
        # Calculate moving averages
        for period in ma_periods:
            if len(data) >= period:
                ma = data["close"].rolling(window=period).mean()
                
                # Get the latest MA value
                latest_ma = ma.iloc[-1]
                
                # Determine if it's support or resistance
                current_price = data["close"].iloc[-1]
                level_type = "resistance" if latest_ma > current_price else "support"
                
                # Calculate strength based on period (longer periods are stronger)
                strength = period / max(ma_periods) * 100
                
                levels.append({
                    "price": float(latest_ma),
                    "type": level_type,
                    "strength": float(strength),
                    "method": SupportResistanceMethod.MOVING_AVERAGE.value,
                    "touches": int(period),  # Use period as touches
                    "last_touch_date": data.index[-1].isoformat() if hasattr(data.index[-1], 'isoformat') else str(data.index[-1]),
                    "metadata": {
                        "period": period
                    }
                })
                
        return levels
        
    def _identify_levels_fibonacci(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify support and resistance levels using Fibonacci retracement method.
        
        Args:
            data: Market data
            parameters: Additional parameters
            
        Returns:
            List of identified levels
        """
        levels = []
        
        # Get parameters
        lookback = parameters.get("lookback", 100)
        
        # Ensure we have enough data
        if len(data) < lookback:
            return levels
            
        # Get the highest high and lowest low in the lookback period
        high = data["high"].iloc[-lookback:].max()
        low = data["low"].iloc[-lookback:].min()
        
        # Calculate Fibonacci levels
        fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        
        for fib in fib_levels:
            level = high - (high - low) * fib
            
            # Determine if it's support or resistance
            current_price = data["close"].iloc[-1]
            level_type = "resistance" if level > current_price else "support"
            
            # Calculate strength based on Fibonacci level
            strength = (1 - abs(fib - 0.5)) * 100
            
            levels.append({
                "price": float(level),
                "type": level_type,
                "strength": float(strength),
                "method": SupportResistanceMethod.FIBONACCI.value,
                "touches": 0,  # Fibonacci levels don't have touches
                "last_touch_date": None,
                "metadata": {
                    "fibonacci_level": fib
                }
            })
            
        return levels
        
    def _identify_levels_pivot_points(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify support and resistance levels using pivot points method.
        
        Args:
            data: Market data
            parameters: Additional parameters
            
        Returns:
            List of identified levels
        """
        levels = []
        
        # Get parameters
        pivot_type = parameters.get("pivot_type", "standard")
        
        # Ensure we have enough data
        if len(data) < 1:
            return levels
            
        # Get the previous day's high, low, and close
        prev_high = data["high"].iloc[-2]
        prev_low = data["low"].iloc[-2]
        prev_close = data["close"].iloc[-2]
        
        # Calculate pivot point
        pivot = (prev_high + prev_low + prev_close) / 3
        
        # Calculate support and resistance levels
        if pivot_type == "standard":
            r1 = 2 * pivot - prev_low
            r2 = pivot + (prev_high - prev_low)
            r3 = r1 + (prev_high - prev_low)
            
            s1 = 2 * pivot - prev_high
            s2 = pivot - (prev_high - prev_low)
            s3 = s1 - (prev_high - prev_low)
            
            resistance_levels = [
                {"level": r1, "name": "R1", "strength": 90},
                {"level": r2, "name": "R2", "strength": 80},
                {"level": r3, "name": "R3", "strength": 70}
            ]
            
            support_levels = [
                {"level": s1, "name": "S1", "strength": 90},
                {"level": s2, "name": "S2", "strength": 80},
                {"level": s3, "name": "S3", "strength": 70}
            ]
            
        elif pivot_type == "fibonacci":
            r1 = pivot + 0.382 * (prev_high - prev_low)
            r2 = pivot + 0.618 * (prev_high - prev_low)
            r3 = pivot + 1.0 * (prev_high - prev_low)
            
            s1 = pivot - 0.382 * (prev_high - prev_low)
            s2 = pivot - 0.618 * (prev_high - prev_low)
            s3 = pivot - 1.0 * (prev_high - prev_low)
            
            resistance_levels = [
                {"level": r1, "name": "R1 (0.382)", "strength": 90},
                {"level": r2, "name": "R2 (0.618)", "strength": 80},
                {"level": r3, "name": "R3 (1.0)", "strength": 70}
            ]
            
            support_levels = [
                {"level": s1, "name": "S1 (0.382)", "strength": 90},
                {"level": s2, "name": "S2 (0.618)", "strength": 80},
                {"level": s3, "name": "S3 (1.0)", "strength": 70}
            ]
            
        else:
            return levels
            
        # Add pivot point
        current_price = data["close"].iloc[-1]
        pivot_type = "resistance" if pivot > current_price else "support"
        
        levels.append({
            "price": float(pivot),
            "type": pivot_type,
            "strength": float(100),  # Pivot point has highest strength
            "method": SupportResistanceMethod.PIVOT_POINTS.value,
            "touches": 0,  # Pivot points don't have touches
            "last_touch_date": None,
            "metadata": {
                "name": "Pivot",
                "pivot_type": pivot_type
            }
        })
        
        # Add resistance levels
        for r in resistance_levels:
            levels.append({
                "price": float(r["level"]),
                "type": "resistance",
                "strength": float(r["strength"]),
                "method": SupportResistanceMethod.PIVOT_POINTS.value,
                "touches": 0,  # Pivot points don't have touches
                "last_touch_date": None,
                "metadata": {
                    "name": r["name"],
                    "pivot_type": pivot_type
                }
            })
            
        # Add support levels
        for s in support_levels:
            levels.append({
                "price": float(s["level"]),
                "type": "support",
                "strength": float(s["strength"]),
                "method": SupportResistanceMethod.PIVOT_POINTS.value,
                "touches": 0,  # Pivot points don't have touches
                "last_touch_date": None,
                "metadata": {
                    "name": s["name"],
                    "pivot_type": pivot_type
                }
            })
            
        return levels
        
    def _identify_levels_volume_profile(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify support and resistance levels using volume profile method.
        
        Args:
            data: Market data
            parameters: Additional parameters
            
        Returns:
            List of identified levels
        """
        levels = []
        
        # Get parameters
        num_bins = parameters.get("num_bins", 20)
        
        # Ensure we have enough data and volume
        if len(data) < 10 or "volume" not in data.columns:
            return levels
            
        # Create price bins
        price_min = data["low"].min()
        price_max = data["high"].max()
        bin_size = (price_max - price_min) / num_bins
        
        bins = np.linspace(price_min, price_max, num_bins + 1)
        
        # Calculate volume profile
        volume_profile = np.zeros(num_bins)
        
        for i in range(len(data)):
            # Get the typical price for the bar
            typical_price = (data["high"].iloc[i] + data["low"].iloc[i] + data["close"].iloc[i]) / 3
            
            # Find the bin index
            bin_idx = int((typical_price - price_min) / bin_size)
            
            # Ensure bin index is valid
            bin_idx = max(0, min(bin_idx, num_bins - 1))
            
            # Add volume to the bin
            volume_profile[bin_idx] += data["volume"].iloc[i]
            
        # Find local maxima in volume profile (high volume nodes)
        local_maxima = []
        
        for i in range(1, num_bins - 1):
            if volume_profile[i] > volume_profile[i - 1] and volume_profile[i] > volume_profile[i + 1]:
                local_maxima.append(i)
                
        # Create levels from high volume nodes
        for idx in local_maxima:
            price = price_min + (idx + 0.5) * bin_size
            
            # Determine if it's support or resistance
            current_price = data["close"].iloc[-1]
            level_type = "resistance" if price > current_price else "support"
            
            # Calculate strength based on volume
            strength = volume_profile[idx] / volume_profile.max() * 100
            
            levels.append({
                "price": float(price),
                "type": level_type,
                "strength": float(strength),
                "method": SupportResistanceMethod.VOLUME_PROFILE.value,
                "touches": 0,  # Volume profile doesn't have touches
                "last_touch_date": None,
                "metadata": {
                    "volume": float(volume_profile[idx])
                }
            })
            
        return levels
        
    def _identify_levels_fractal(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify support and resistance levels using fractal method.
        
        Args:
            data: Market data
            parameters: Additional parameters
            
        Returns:
            List of identified levels
        """
        levels = []
        
        # Get parameters
        window = parameters.get("window", 2)
        
        # Ensure we have enough data
        if len(data) < 2 * window + 1:
            return levels
            
        # Find fractal patterns
        for i in range(window, len(data) - window):
            # Check for bullish fractal (support)
            is_bullish_fractal = True
            for j in range(1, window + 1):
                if data["low"].iloc[i] >= data["low"].iloc[i - j] or data["low"].iloc[i] >= data["low"].iloc[i + j]:
                    is_bullish_fractal = False
                    break
                    
            # Check for bearish fractal (resistance)
            is_bearish_fractal = True
            for j in range(1, window + 1):
                if data["high"].iloc[i] <= data["high"].iloc[i - j] or data["high"].iloc[i] <= data["high"].iloc[i + j]:
                    is_bearish_fractal = False
                    break
                    
            # Add bullish fractal (support)
            if is_bullish_fractal:
                levels.append({
                    "price": float(data["low"].iloc[i]),
                    "type": "support",
                    "strength": float(50),  # Default strength
                    "method": SupportResistanceMethod.FRACTAL.value,
                    "touches": 1,  # Fractals have one touch by definition
                    "last_touch_date": data.index[i].isoformat() if hasattr(data.index[i], 'isoformat') else str(data.index[i])
                })
                
            # Add bearish fractal (resistance)
            if is_bearish_fractal:
                levels.append({
                    "price": float(data["high"].iloc[i]),
                    "type": "resistance",
                    "strength": float(50),  # Default strength
                    "method": SupportResistanceMethod.FRACTAL.value,
                    "touches": 1,  # Fractals have one touch by definition
                    "last_touch_date": data.index[i].isoformat() if hasattr(data.index[i], 'isoformat') else str(data.index[i])
                })
                
        return levels
        
    def _identify_levels_custom(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify support and resistance levels using custom method.
        
        Args:
            data: Market data
            parameters: Custom parameters
            
        Returns:
            List of identified levels
        """
        # Custom method implementation would go here
        return []
        
    def _find_local_maxima(
        self,
        data: np.ndarray,
        window: int = 5
    ) -> List[int]:
        """
        Find local maxima in data.
        
        Args:
            data: Data to find local maxima in
            window: Window size for local maxima detection
            
        Returns:
            Indices of local maxima
        """
        local_maxima = []
        
        for i in range(window, len(data) - window):
            is_local_max = True
            
            for j in range(1, window + 1):
                if data[i] <= data[i - j] or data[i] <= data[i + j]:
                    is_local_max = False
                    break
                    
            if is_local_max:
                local_maxima.append(i)
                
        return local_maxima
        
    def _find_local_minima(
        self,
        data: np.ndarray,
        window: int = 5
    ) -> List[int]:
        """
        Find local minima in data.
        
        Args:
            data: Data to find local minima in
            window: Window size for local minima detection
            
        Returns:
            Indices of local minima
        """
        local_minima = []
        
        for i in range(window, len(data) - window):
            is_local_min = True
            
            for j in range(1, window + 1):
                if data[i] >= data[i - j] or data[i] >= data[i + j]:
                    is_local_min = False
                    break
                    
            if is_local_min:
                local_minima.append(i)
                
        return local_minima
        
    def _group_similar_levels(
        self,
        data: np.ndarray,
        indices: List[int],
        window: int = 5
    ) -> Dict[float, int]:
        """
        Group similar price levels and count touches.
        
        Args:
            data: Price data
            indices: Indices of local extrema
            window: Window size for grouping
            
        Returns:
            Dictionary mapping price levels to touch counts
        """
        levels = {}
        
        for idx in indices:
            price = data[idx]
            
            # Check if the price is close to an existing level
            found_match = False
            
            for level in list(levels.keys()):
                if abs(price - level) / level < 0.01:  # Within 1% of the level
                    levels[level] += 1
                    found_match = True
                    break
                    
            # If no match found, add a new level
            if not found_match:
                levels[price] = 1
                
        return levels
        
    def _remove_duplicate_levels(
        self,
        levels: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate levels (within a small range).
        
        Args:
            levels: List of levels
            
        Returns:
            List of unique levels
        """
        if not levels:
            return []
            
        # Sort levels by price
        sorted_levels = sorted(levels, key=lambda x: x["price"])
        
        unique_levels = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # Check if the level is close to the last unique level
            last_level = unique_levels[-1]
            
            if abs(level["price"] - last_level["price"]) / last_level["price"] < 0.01:  # Within 1% of the last level
                # Keep the stronger level
                if level["strength"] > last_level["strength"]:
                    unique_levels[-1] = level
            else:
                unique_levels.append(level)
                
        return unique_levels