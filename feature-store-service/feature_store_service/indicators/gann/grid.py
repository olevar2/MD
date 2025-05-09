"""
Gann Grid Module.

This module provides implementation of Gann grid.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from feature_store_service.indicators.gann.base import BaseGannIndicator
from feature_store_service.utils.swing_points import find_swing_highs, find_swing_lows  # Assuming utility exists


class GannGrid(BaseGannIndicator):
    """
    Gann Grid

    Overlays a grid on the chart based on a significant pivot point.
    The grid lines are drawn at specific price and time intervals derived
    from the pivot, often using Gann angles or fixed increments.
    Helps identify potential support/resistance and time turning points.
    """

    def __init__(
        self,
        pivot_type: str = "swing_low",
        lookback_period: int = 100,
        price_interval: Optional[float] = None,  # e.g., price units per grid line
        time_interval: Optional[int] = None,     # e.g., bars per grid line
        auto_interval: bool = True,  # Automatically determine intervals based on volatility/range
        num_price_lines: int = 5,  # Number of lines above/below pivot
        num_time_lines: int = 10,  # Number of lines forward/backward from pivot
        price_scaling: float = 1.0,  # Optional scaling for price intervals if auto-calculating
        **kwargs
    ):
        """
        Initialize Gann Grid indicator.

        Args:
            pivot_type: Type of pivot to use ('swing_low', 'swing_high')
            lookback_period: Number of bars to look back for finding pivot
            price_interval: Fixed price distance between horizontal grid lines
            time_interval: Fixed time distance (bars) between vertical grid lines
            auto_interval: If True, calculate intervals based on recent price range/volatility
            num_price_lines: Number of horizontal lines above and below the pivot
            num_time_lines: Number of vertical lines forward and backward from the pivot
            price_scaling: Scaling factor for price intervals if auto-calculating
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.pivot_type = pivot_type
        self.lookback_period = lookback_period
        self.price_interval = price_interval
        self.time_interval = time_interval
        self.auto_interval = auto_interval
        self.num_price_lines = num_price_lines
        self.num_time_lines = num_time_lines
        self.price_scaling = price_scaling

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gann Grid lines for the given data.
        Adds columns for horizontal and vertical grid line levels/positions.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Gann Grid information
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")

        result = data.copy()

        # Find pivot point
        pivot_idx, pivot_price = self._find_pivot_point(result, self.pivot_type, self.lookback_period)

        if pivot_idx is None or pivot_price is None:
            print("Warning: Could not find pivot for Gann Grid.")
            return result  # Not enough data or pivot not found

        pivot_pos = result.index.get_loc(pivot_idx)

        # Determine price and time intervals
        p_interval, t_interval = self._determine_intervals(result, pivot_price)

        if p_interval <= 0 or t_interval <= 0:
            print("Warning: Invalid intervals calculated for Gann Grid.")
            return result

        # Calculate and store horizontal price grid lines
        for i in range(-self.num_price_lines, self.num_price_lines + 1):
            if i == 0:
                continue  # Skip the pivot line itself
            price_level = pivot_price + (i * p_interval)
            col_name = f"gann_grid_price_{i:+}".replace('+', 'p').replace('-', 'm')
            result[col_name] = price_level

        # Calculate and store vertical time grid lines (as markers)
        result['gann_grid_time_marker'] = 0
        for i in range(-self.num_time_lines, self.num_time_lines + 1):
            if i == 0:
                continue
            time_pos = pivot_pos + (i * t_interval)
            if 0 <= time_pos < len(result):
                time_idx = result.index[time_pos]
                result.loc[time_idx, 'gann_grid_time_marker'] = i  # Mark with relative interval number
                # Add specific marker column
                col_name = f'gann_grid_time_{i:+}'.replace('+', 'p').replace('-', 'm')
                if col_name not in result.columns:
                    result[col_name] = False
                result.loc[time_idx, col_name] = True

        # Add pivot marker
        result['gann_grid_pivot_idx'] = False
        result.loc[pivot_idx, 'gann_grid_pivot_idx'] = True
        result['gann_grid_pivot_price'] = pivot_price
        result['gann_grid_price_interval'] = p_interval
        result['gann_grid_time_interval'] = t_interval

        return result

    def _determine_intervals(self, data: pd.DataFrame, pivot_price: float) -> Tuple[float, int]:
        """Determine price and time intervals for the grid."""
        p_interval = self.price_interval
        t_interval = self.time_interval

        if self.auto_interval:
            lookback_data = data.iloc[-self.lookback_period:] if len(data) > self.lookback_period else data
            if len(lookback_data) > 1:
                price_range = lookback_data['high'].max() - lookback_data['low'].min()
                avg_daily_range = (lookback_data['high'] - lookback_data['low']).mean()

                # Heuristic for price interval (e.g., fraction of range or ATR)
                if p_interval is None:
                    # Use a fraction of the average range, scaled
                    p_interval = (avg_daily_range / 4) * self.price_scaling if avg_daily_range > 0 else (pivot_price * 0.01) * self.price_scaling
                    if p_interval <= 0:
                        p_interval = pivot_price * 0.01  # Fallback

                # Heuristic for time interval (e.g., related to price interval via 1x1 concept or fixed)
                if t_interval is None:
                    # Try to relate to price interval via scaling (Gann 1x1 idea)
                    # If price_scaling represents price units per bar for 1x1 angle
                    # Then time_interval * price_scaling = price_interval
                    if self.price_scaling > 0:
                        t_interval_calc = p_interval / self.price_scaling
                        t_interval = max(1, int(round(t_interval_calc)))  # Ensure at least 1 bar
                    else:
                        t_interval = 10  # Default fixed interval if scaling is zero
            else:
                # Fallback if not enough data for auto calculation
                if p_interval is None:
                    p_interval = pivot_price * 0.01
                if t_interval is None:
                    t_interval = 10
        else:
            # Use fixed intervals if provided, with defaults if missing
            if p_interval is None:
                p_interval = pivot_price * 0.01
            if t_interval is None:
                t_interval = 10

        # Ensure intervals are positive
        p_interval = max(p_interval, 1e-9)  # Avoid zero or negative price interval
        t_interval = max(t_interval, 1)     # Ensure time interval is at least 1

        return p_interval, t_interval

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Gann Grid',
            'description': 'Overlays a price/time grid based on a pivot point',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'pivot_type',
                    'description': 'Type of pivot point to use',
                    'type': 'string',
                    'default': 'swing_low',
                    'options': ['swing_low', 'swing_high']
                },
                {
                    'name': 'lookback_period',
                    'description': 'Number of bars to look back for finding pivot',
                    'type': 'int',
                    'default': 100
                },
                {
                    'name': 'price_interval',
                    'description': 'Fixed price distance between horizontal lines (optional)',
                    'type': 'float',
                    'default': None
                },
                {
                    'name': 'time_interval',
                    'description': 'Fixed time distance (bars) between vertical lines (optional)',
                    'type': 'int',
                    'default': None
                },
                {
                    'name': 'auto_interval',
                    'description': 'Automatically determine intervals based on volatility/range',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'num_price_lines',
                    'description': 'Number of horizontal lines above/below pivot',
                    'type': 'int',
                    'default': 5
                },
                {
                    'name': 'num_time_lines',
                    'description': 'Number of vertical lines forward/backward from pivot',
                    'type': 'int',
                    'default': 10
                },
                {
                    'name': 'price_scaling',
                    'description': 'Scaling factor for auto price interval calculation',
                    'type': 'float',
                    'default': 1.0
                }
            ]
        }
