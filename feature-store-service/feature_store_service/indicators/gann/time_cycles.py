"""
Gann Time Cycles Module.

This module provides implementation of Gann time cycles.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from feature_store_service.indicators.gann.base import BaseGannIndicator


class GannTimeCycles(BaseGannIndicator):
    """
    Gann Time Cycles

    Identifies potential future turning points in time based on Gann's cycle theories,
    often using significant past highs or lows as starting points and projecting
    forward using specific time intervals (e.g., 90, 180, 360 days/bars).
    """

    def __init__(
        self,
        cycle_lengths: Optional[List[int]] = None,
        starting_point_type: str = "major_low",
        lookback_period: int = 200,
        auto_detect_start: bool = True,
        max_cycles: int = 5,
        **kwargs
    ):
        """
        Initialize Gann Time Cycles indicator.

        Args:
            cycle_lengths: List of cycle lengths in bars (e.g., [90, 180, 360])
            starting_point_type: Type of pivot to use as start ('major_low', 'major_high')
            lookback_period: Number of bars to look back for finding starting point
            auto_detect_start: Whether to automatically detect the starting point
            max_cycles: Maximum number of cycle projections for each length
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        # Default cycle lengths based on common Gann cycles (can be adapted for timeframe)
        self.cycle_lengths = cycle_lengths or [30, 60, 90, 120, 180, 270, 360]
        self.starting_point_type = starting_point_type
        self.lookback_period = lookback_period
        self.auto_detect_start = auto_detect_start
        self.max_cycles = max_cycles

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gann Time Cycle points for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Gann Time Cycle markers
        """
        required_cols = ['high', 'low']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Find starting point
        if self.auto_detect_start:
            start_pos = self._detect_starting_point(result)
        else:
            # Manual start point needs to be provided via kwargs or other means
            # For now, default to a point within the lookback period
            start_pos = max(0, len(result) - self.lookback_period)

        if start_pos is None:
            return result

        start_idx = data.index[start_pos]

        # Create columns for time cycle markers
        result['gann_time_cycle_point'] = 0
        result['gann_time_cycle_start'] = False
        result.loc[start_idx, 'gann_time_cycle_start'] = True

        # Project time cycles forward
        cycle_counter = 1
        for length in self.cycle_lengths:
            for i in range(1, self.max_cycles + 1):
                # Initialize column for this cycle even if no points are found
                col_name = f'gann_time_cycle_{length}_{i}'
                if col_name not in result.columns:
                    result[col_name] = False

                cycle_pos = start_pos + (length * i)
                if cycle_pos < len(result):
                    cycle_idx = result.index[cycle_pos]
                    # Mark the cycle point, potentially overwriting if multiple cycles align
                    result.loc[cycle_idx, 'gann_time_cycle_point'] = cycle_counter
                    result.loc[cycle_idx, col_name] = True

            cycle_counter += 1 # Increment counter for next cycle length group

        return result

    def _detect_starting_point(self, data: pd.DataFrame) -> Optional[int]:
        """
        Detect a significant starting point (major high or low).

        Returns:
            Position (index) of the starting point in the data
        """
        lookback_data = data.iloc[-self.lookback_period:] if len(data) > self.lookback_period else data

        if lookback_data.empty:
            return None

        start_pos = None
        if self.starting_point_type == "major_low":
            low_idx = lookback_data['low'].idxmin()
            start_pos = data.index.get_loc(low_idx)
        elif self.starting_point_type == "major_high":
            high_idx = lookback_data['high'].idxmax()
            start_pos = data.index.get_loc(high_idx)
        else:
            # Default to major low
            low_idx = lookback_data['low'].idxmin()
            start_pos = data.index.get_loc(low_idx)

        return start_pos

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Gann Time Cycles',
            'description': 'Identifies potential future turning points based on Gann cycle lengths',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'cycle_lengths',
                    'description': 'List of cycle lengths in bars',
                    'type': 'list',
                    'default': [30, 60, 90, 120, 180, 270, 360]
                },
                {
                    'name': 'starting_point_type',
                    'description': 'Type of pivot to use as start',
                    'type': 'string',
                    'default': 'major_low',
                    'options': ['major_low', 'major_high']
                },
                {
                    'name': 'lookback_period',
                    'description': 'Number of bars to look back for finding starting point',
                    'type': 'int',
                    'default': 200
                },
                {
                    'name': 'auto_detect_start',
                    'description': 'Whether to automatically detect the starting point',
                    'type': 'bool',
                    'default': True
                },
                {
                    'name': 'max_cycles',
                    'description': 'Maximum number of cycle projections for each length',
                    'type': 'int',
                    'default': 5
                }
            ]
        }
