"""
Gann Square of 144 Module.

This module provides implementation of Gann Square of 144.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import math

from feature_store_service.indicators.gann.base import BaseGannIndicator


class GannSquare144(BaseGannIndicator):
    """
    Gann Square of 144

    Calculates potential support and resistance levels based on the Square of 144 concept.
    This often relates to a 144-day or 144-bar cycle, or uses increments derived
    from the square root of 144 (which is 12). This implementation uses the
    standard square root method but allows adjusting the base increment.
    """

    def __init__(
        self,
        pivot_type: str = "major_low",  # 'major_low', 'major_high', 'manual'
        manual_pivot_price: Optional[float] = None,
        manual_pivot_time_idx: Optional[int] = None,  # Position index
        lookback_period: int = 200,  # Longer lookback might be relevant for 144 cycle
        auto_detect_pivot: bool = True,
        num_levels: int = 4,
        base_increment: float = 0.125,  # Standard Sq9 increment, adjust if 144 implies different scaling
        use_time_squaring: bool = False,  # Optionally add time squaring
        time_price_ratio: float = 1.0,  # Required if use_time_squaring is True
        **kwargs
    ):
        """
        Initialize Gann Square of 144 indicator.

        Args:
            pivot_type: Type of pivot ('major_low', 'major_high', 'manual')
            manual_pivot_price: Manual pivot price (if pivot_type='manual')
            manual_pivot_time_idx: Manual pivot time position index (if pivot_type='manual')
            lookback_period: Bars for auto-detecting pivot (consider >= 144)
            auto_detect_pivot: Whether to automatically detect pivot point
            num_levels: Number of square levels to calculate outwards
            base_increment: The fundamental increment added/subtracted from sqrt(price).
                            Default is 0.125 (Sq9 45deg). Adjust based on specific Sq144 interpretation.
            use_time_squaring: If True, calculate time turning points as well.
            time_price_ratio: Scaling factor for time squaring (Price units per Time unit).
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.pivot_type = pivot_type
        self.manual_pivot_price = manual_pivot_price
        self.manual_pivot_time_idx = manual_pivot_time_idx
        self.lookback_period = lookback_period
        self.auto_detect_pivot = auto_detect_pivot
        self.num_levels = num_levels
        self.base_increment = base_increment  # Key parameter for Sq144 interpretation
        self.use_time_squaring = use_time_squaring
        self.time_price_ratio = time_price_ratio

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gann Square of 144 price levels (and optionally time points).

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Gann Square 144 levels and optional time markers
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")

        result = data.copy()

        # Determine pivot point (price and time position)
        pivot_pos, pivot_price = self._determine_pivot(result, self.pivot_type, self.lookback_period, 
                                                      self.auto_detect_pivot, self.manual_pivot_price, 
                                                      self.manual_pivot_time_idx)

        if pivot_pos is None or pivot_price is None or pivot_price <= 0:
            print("Warning: Invalid pivot point for Gann Square 144.")
            return result

        pivot_idx_ts = result.index[pivot_pos]
        sqrt_pivot_price = math.sqrt(pivot_price)

        # --- Calculate Price Levels ---
        for i in range(1, self.num_levels + 1):
            increment = self.base_increment * i

            # Support levels
            level_down = (sqrt_pivot_price - increment)**2
            # Resistance levels
            level_up = (sqrt_pivot_price + increment)**2

            # Add levels to DataFrame
            result[f'gann_sq144_sup_{i}'] = level_down if level_down > 0 else np.nan
            result[f'gann_sq144_res_{i}'] = level_up

        # --- Optionally Calculate Time Turning Points ---
        if self.use_time_squaring:
            result['gann_sq144_time_marker'] = 0
            if self.time_price_ratio > 0:
                for i in range(1, self.num_levels + 1):  # Use num_levels for time too
                    increment = self.base_increment * i

                    # Time points derived from resistance levels
                    price_level_up = (sqrt_pivot_price + increment)**2
                    time_offset_up = int(round(price_level_up / self.time_price_ratio))
                    time_pos_up_fwd = pivot_pos + time_offset_up
                    time_pos_up_bwd = pivot_pos - time_offset_up

                    # Time points derived from support levels
                    price_level_down = (sqrt_pivot_price - increment)**2
                    if price_level_down > 0:
                        time_offset_down = int(round(price_level_down / self.time_price_ratio))
                        time_pos_down_fwd = pivot_pos + time_offset_down
                        time_pos_down_bwd = pivot_pos - time_offset_down
                    else:
                        time_pos_down_fwd, time_pos_down_bwd = None, None

                    # Mark time points
                    for time_pos, marker_val in [
                        (time_pos_up_fwd, i), (time_pos_up_bwd, -i),
                        (time_pos_down_fwd, i), (time_pos_down_bwd, -i)
                    ]:
                        if time_pos is not None and 0 <= time_pos < len(result):
                            time_idx = result.index[time_pos]
                            result.loc[time_idx, 'gann_sq144_time_marker'] = marker_val
                            # Add specific marker column
                            direction = 'fwd' if marker_val > 0 else 'bwd'
                            level_type = 'res' if time_pos in [time_pos_up_fwd, time_pos_up_bwd] else 'sup'
                            col_name = f'gann_sq144_time_{level_type}_{abs(marker_val)}_{direction}'
                            if col_name not in result.columns:
                                result[col_name] = False
                            result.loc[time_idx, col_name] = True
            else:
                print("Warning: time_price_ratio must be positive for time squaring.")

        # Add pivot marker
        result['gann_sq144_pivot_idx'] = False
        result.loc[pivot_idx_ts, 'gann_sq144_pivot_idx'] = True
        result['gann_sq144_pivot_price'] = pivot_price
        result['gann_sq144_pivot_pos'] = pivot_pos

        return result

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Gann Square 144',
            'description': 'Calculates Gann Square levels, potentially related to 144 cycle/scaling',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'pivot_type',
                    'description': 'Type of pivot point to use',
                    'type': 'string',
                    'default': 'major_low',
                    'options': ['major_low', 'major_high', 'manual']
                },
                {
                    'name': 'manual_pivot_price',
                    'description': 'Manual pivot price (if pivot_type=\'manual\')',
                    'type': 'float',
                    'default': None
                },
                {
                    'name': 'manual_pivot_time_idx',
                    'description': 'Manual pivot time position index (if pivot_type=\'manual\')',
                    'type': 'int',
                    'default': None
                },
                {
                    'name': 'lookback_period',
                    'description': 'Bars for auto-detecting pivot (e.g., >= 144)',
                    'type': 'int',
                    'default': 200
                },
                {
                    'name': 'auto_detect_pivot',
                    'description': 'Whether to automatically detect pivot point',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'num_levels',
                    'description': 'Number of square levels outwards',
                    'type': 'int',
                    'default': 4
                },
                {
                    'name': 'base_increment',
                    'description': 'Sqrt increment for levels (adjust for Sq144 interpretation)',
                    'type': 'float',
                    'default': 0.125
                },
                {
                    'name': 'use_time_squaring',
                    'description': 'Calculate time turning points based on square levels',
                    'type': 'bool',
                    'default': False
                },
                {
                    'name': 'time_price_ratio',
                    'description': 'Scaling factor for time squaring (Price units per Time unit)',
                    'type': 'float',
                    'default': 1.0  # Needs calibration if use_time_squaring=True
                }
            ]
        }
