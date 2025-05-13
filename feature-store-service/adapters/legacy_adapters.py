"""
Legacy Adapters for Gann Tools.

This module provides adapter classes for backward compatibility with the original gann_tools.py module.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union

from core.angles import GannAngles as NewGannAngles
from core.fans_1 import GannFan as NewGannFan
from core.square_of_9 import GannSquare as NewGannSquare


class GannAngles:
    """
    Legacy adapter for GannAngles.

    This class provides backward compatibility with the original GannAngles class.
    """

    def __init__(
        self,
        pivot_price: Optional[float] = None,
        pivot_date: Optional[datetime] = None,
        auto_detect_pivot: bool = False,
        angle_types: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize GannAngles with legacy parameters.
        """
        self.pivot_price = pivot_price
        self.pivot_date = pivot_date
        self.auto_detect_pivot = auto_detect_pivot
        self.angle_types = angle_types or ["1x8", "1x4", "1x3", "1x2", "1x1", "2x1", "3x1", "4x1", "8x1"]
        self.kwargs = kwargs

    def calculate_angles(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Calculate Gann angles from pivot point.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary of angle names and their values
        """
        # Adapt to new implementation
        kwargs = self.kwargs.copy()
        if 'pivot_type' not in kwargs:
            kwargs['pivot_type'] = "manual" if self.pivot_price is not None else "swing_low"
        if 'angle_types' not in kwargs:
            kwargs['angle_types'] = self.angle_types
        if 'lookback_period' not in kwargs:
            kwargs['lookback_period'] = 100
        if 'price_scaling' not in kwargs:
            kwargs['price_scaling'] = 1.0
        if 'projection_bars' not in kwargs:
            kwargs['projection_bars'] = 50

        new_gann_angles = NewGannAngles(**kwargs)

        # Find pivot position if pivot_date is provided
        pivot_pos = None
        if self.pivot_date is not None:
            for i, date in enumerate(data.index):
                if date == self.pivot_date:
                    pivot_pos = i
                    break

        # Calculate using new implementation
        result = new_gann_angles.calculate(data)

        # Convert to legacy format
        angles_dict = {}
        for angle_type in self.angle_types:
            up_col = f"gann_angle_up_{angle_type}"
            if up_col in result.columns:
                angles_dict[angle_type] = result[up_col].tolist()

        return angles_dict


class GannSquare9:
    """
    Legacy adapter for GannSquare9.

    This class provides backward compatibility with the original GannSquare9 class.
    """

    def __init__(
        self,
        base_price: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize GannSquare9 with legacy parameters.
        """
        self.base_price = base_price
        self.kwargs = kwargs

    def calculate_levels(self, n_levels: int = 5) -> List[List[float]]:
        """
        Calculate Square of 9 levels.

        Args:
            n_levels: Number of levels to calculate

        Returns:
            List of levels, each containing price points
        """
        # Create dummy data for the new implementation
        dates = [datetime.now() - timedelta(days=i) for i in range(100)]
        dates.reverse()
        data = pd.DataFrame({
            'open': [100] * 100,
            'high': [110] * 100,
            'low': [90] * 100,
            'close': [105] * 100,
            'volume': [1000] * 100
        }, index=dates)

        # Adapt to new implementation
        new_gann_square = NewGannSquare(
            square_type="square_of_9",
            pivot_price=self.base_price,
            auto_detect_pivot=False if self.base_price is not None else True,
            num_levels=n_levels,
            **self.kwargs
        )

        # Calculate using new implementation
        result = new_gann_square.calculate(data)

        # Convert to legacy format
        levels = []
        for i in range(1, n_levels + 1):
            level = []
            for angle in [45, 90, 135, 180]:
                sup_col = f"gann_sq_sup_{angle}_{i}"
                res_col = f"gann_sq_res_{angle}_{i}"
                if sup_col in result.columns:
                    level.append(result[sup_col].iloc[0])
                if res_col in result.columns:
                    level.append(result[res_col].iloc[0])
            levels.append(level)

        # Add base price to first level
        if self.base_price is not None and levels:
            levels[0].append(self.base_price)

        return levels

    def get_cardinal_levels(self, n_revolutions: int = 3) -> List[float]:
        """
        Get cardinal direction levels (90-degree increments).

        Args:
            n_revolutions: Number of revolutions around the square

        Returns:
            List of cardinal direction levels
        """
        # Simplified implementation for backward compatibility
        if self.base_price is None:
            return [100 + i * 10 for i in range(n_revolutions * 4)]

        sqrt_base = np.sqrt(self.base_price)
        cardinals = []

        for i in range(1, n_revolutions + 1):
            # North (90 degrees)
            cardinals.append((sqrt_base + i * 0.25)**2)
            # East (0 degrees)
            cardinals.append((sqrt_base + i * 0.5)**2)
            # South (270 degrees)
            cardinals.append((sqrt_base + i * 0.75)**2)
            # West (180 degrees)
            cardinals.append((sqrt_base + i * 1.0)**2)

        return cardinals

    def get_support_resistance_levels(self, current_price: float, n_levels: int = 3) -> Dict[str, List[float]]:
        """
        Get support and resistance levels for a given price.

        Args:
            current_price: Current price to find levels around
            n_levels: Number of levels to find in each direction

        Returns:
            Dictionary with 'support' and 'resistance' lists
        """
        # Simplified implementation for backward compatibility
        sqrt_current = np.sqrt(current_price)

        support = []
        resistance = []

        for i in range(1, n_levels + 1):
            # Support levels
            support.append((sqrt_current - i * 0.125)**2)
            # Resistance levels
            resistance.append((sqrt_current + i * 0.125)**2)

        return {
            'support': support,
            'resistance': resistance
        }


class GannFan:
    """
    Legacy adapter for GannFan.

    This class provides backward compatibility with the original GannFan class.
    """

    def __init__(
        self,
        pivot_price: Optional[float] = None,
        pivot_date: Optional[datetime] = None,
        auto_detect_pivot: bool = False,
        fan_angles: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize GannFan with legacy parameters.
        """
        self.pivot_price = pivot_price
        self.pivot_date = pivot_date
        self.auto_detect_pivot = auto_detect_pivot
        self.fan_angles = fan_angles or ["1x8", "1x4", "1x3", "1x2", "1x1", "2x1", "3x1", "4x1", "8x1"]
        self.kwargs = kwargs

    def calculate_fan(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Calculate Gann Fan lines.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary of fan line names and their values
        """
        # Adapt to new implementation
        new_gann_fan = NewGannFan(
            pivot_type="manual" if self.pivot_price is not None else "swing_low",
            fan_angles=self.fan_angles,
            lookback_period=100,
            price_scaling=1.0,
            projection_bars=50,
            **self.kwargs
        )

        # Create a modified copy of the data with the pivot price at the pivot date
        modified_data = data.copy()

        # Find pivot position if pivot_date is provided
        pivot_pos = None
        if self.pivot_date is not None:
            for i, date in enumerate(data.index):
                if date == self.pivot_date:
                    pivot_pos = i
                    break

        # Create a simple implementation for backward compatibility
        fan_dict = {}

        # Define the angles and their ratios
        angle_ratios = {
            "1x8": 1/8, "1x4": 1/4, "1x3": 1/3, "1x2": 1/2, "1x1": 1,
            "2x1": 2, "3x1": 3, "4x1": 4, "8x1": 8
        }

        # If we have a valid pivot point, calculate fan lines
        if pivot_pos is not None and self.pivot_price is not None:
            for angle_type in self.fan_angles:
                if angle_type in angle_ratios:
                    ratio = angle_ratios[angle_type]
                    fan_line = []

                    for i in range(len(data)):
                        time_diff = i - pivot_pos
                        if time_diff >= 0:
                            # Calculate fan line value
                            fan_value = self.pivot_price + (ratio * time_diff)
                            fan_line.append(fan_value)
                        else:
                            # Before pivot point, use None or extrapolate backward
                            fan_line.append(None)

                    fan_dict[angle_type] = fan_line
        else:
            # Fallback to new implementation
            result = new_gann_fan.calculate(data)

            # Convert to legacy format
            for angle_type in self.fan_angles:
                col = f"gann_fan_{angle_type}"
                if col in result.columns:
                    fan_dict[angle_type] = result[col].tolist()

        return fan_dict
