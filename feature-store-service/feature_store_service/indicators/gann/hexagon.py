"""
Gann Hexagon Module.

This module provides implementation of Gann hexagon.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import math

from feature_store_service.indicators.gann.base import BaseGannIndicator


class GannHexagon(BaseGannIndicator):
    """
    Gann Hexagon

    Calculates potential support, resistance, and time turning points based on
    a hexagonal geometric structure. This is a more advanced Gann tool that
    incorporates both price and time elements in a 60-degree-based system.
    """

    def __init__(
        self,
        pivot_type: str = "major_low",  # 'major_low', 'major_high', 'manual'
        manual_pivot_price: Optional[float] = None,
        manual_pivot_time_idx: Optional[int] = None,  # Position index
        lookback_period: int = 200,
        auto_detect_pivot: bool = True,
        degrees: List[int] = [60, 120, 180, 240, 300, 360],
        num_cycles: int = 3,
        time_price_ratio: float = 1.0,  # Price units per Time unit (bar)
        units_per_cycle: float = 360.0,
        **kwargs
    ):
        """
        Initialize Gann Hexagon indicator.

        Args:
            pivot_type: Type of pivot ('major_low', 'major_high', 'manual')
            manual_pivot_price: Manual pivot price (if pivot_type='manual')
            manual_pivot_time_idx: Manual pivot time position index (if pivot_type='manual')
            lookback_period: Bars to look back for auto-detecting pivot
            auto_detect_pivot: Whether to automatically detect pivot point
            degrees: List of key angles (degrees) on the hexagon
            num_cycles: Number of full hexagon cycles/rotations
            time_price_ratio: Scaling factor: Price units per Time unit (bar)
            units_per_cycle: Units (e.g., degrees) in one full cycle
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.pivot_type = pivot_type
        self.manual_pivot_price = manual_pivot_price
        self.manual_pivot_time_idx = manual_pivot_time_idx
        self.lookback_period = lookback_period
        self.auto_detect_pivot = auto_detect_pivot
        self.degrees = sorted(degrees)
        self.num_cycles = num_cycles
        self.time_price_ratio = time_price_ratio
        self.units_per_cycle = units_per_cycle

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gann Hexagon price and time points.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Gann Hexagon information
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
            print("Warning: Invalid pivot point for Gann Hexagon.")
            return result

        pivot_idx_ts = result.index[pivot_pos]

        # Initialize hexagon markers
        result['gann_hexagon_price_marker'] = 0
        result['gann_hexagon_time_marker'] = 0

        # Calculate hexagon points for each cycle and degree
        for cycle in range(1, self.num_cycles + 1):
            for degree in self.degrees:
                # Calculate the price level for this hexagon point
                # Use sine function to create hexagonal pattern
                angle_rad = math.radians(degree)
                cycle_factor = cycle / self.num_cycles  # Normalize to 0-1 range
                
                # Price calculation - create a hexagonal pattern
                price_offset = pivot_price * cycle_factor * math.sin(angle_rad)
                price_level = pivot_price + price_offset
                
                # Skip negative prices
                if price_level <= 0:
                    continue
                
                # Add price level to result
                col_name = f'gann_hexagon_price_c{cycle}_d{degree}'
                result[col_name] = price_level
                
                # Calculate time point based on degree and cycle
                # Convert angular position to time position
                time_units = (degree / 360) * self.units_per_cycle * cycle
                time_bars = int(round(time_units / self.time_price_ratio)) if self.time_price_ratio > 0 else 0
                
                # Calculate forward and backward time positions
                time_pos_fwd = pivot_pos + time_bars
                time_pos_bwd = pivot_pos - time_bars
                
                # Mark time points if within data range
                for time_pos, direction in [(time_pos_fwd, 'fwd'), (time_pos_bwd, 'bwd')]:
                    if 0 <= time_pos < len(result):
                        time_idx = result.index[time_pos]
                        # Add specific marker column
                        col_name = f'gann_hexagon_time_c{cycle}_d{degree}_{direction}'
                        if col_name not in result.columns:
                            result[col_name] = False
                        result.loc[time_idx, col_name] = True
                        
                        # Update general marker
                        marker_val = cycle * (1 if direction == 'fwd' else -1)
                        result.loc[time_idx, 'gann_hexagon_time_marker'] = marker_val

        # Add pivot marker
        result['gann_hexagon_pivot_idx'] = False
        result.loc[pivot_idx_ts, 'gann_hexagon_pivot_idx'] = True
        result['gann_hexagon_pivot_price'] = pivot_price
        result['gann_hexagon_pivot_pos'] = pivot_pos

        return result

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Gann Hexagon',
            'description': 'Calculates price and time points based on hexagonal geometry',
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
                    'description': 'Bars to look back for auto-detecting pivot',
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
                    'name': 'degrees',
                    'description': 'List of key angles (degrees) on the hexagon',
                    'type': 'list',
                    'default': [60, 120, 180, 240, 300, 360]
                },
                {
                    'name': 'num_cycles',
                    'description': 'Number of full hexagon cycles/rotations',
                    'type': 'int',
                    'default': 3
                },
                {
                    'name': 'time_price_ratio',
                    'description': 'Scaling factor: Price units per Time unit (bar)',
                    'type': 'float',
                    'default': 1.0  # Needs calibration
                },
                {
                    'name': 'units_per_cycle',
                    'description': 'Units (e.g., degrees) in one full cycle',
                    'type': 'float',
                    'default': 360.0
                }
            ]
        }
