"""
Gann Analysis Base Module.

This module provides base classes and common utilities for Gann analysis tools.
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import math
from feature_store_service.indicators.base_indicator import BaseIndicator
from feature_store_service.utils.swing_points import find_swing_highs, find_swing_lows


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class BaseGannIndicator(BaseIndicator):
    """
    Base class for all Gann indicators.

    This class provides common functionality for Gann analysis tools.
    """
    category = 'gann'

    def __init__(self, **kwargs):
        """Initialize the Gann indicator."""
        super().__init__(**kwargs)

    @with_exception_handling
    def _find_pivot_point(self, data: pd.DataFrame, pivot_type: str,
        lookback_period: int) ->Tuple[Optional[pd.Timestamp], Optional[float]]:
        """
        Find pivot point based on the specified pivot type.

        Args:
            data: DataFrame with OHLCV data
            pivot_type: Type of pivot to use ('swing_low', 'swing_high', 'recent_low', 'recent_high', 'major_low', 'major_high')
            lookback_period: Number of bars to look back for finding pivot

        Returns:
            Tuple of (pivot_index, pivot_price)

        Raises:
            ValueError: If required data is missing or parameters are invalid
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError('Data must be a pandas DataFrame')
        if len(data) == 0:
            raise ValueError('Data DataFrame is empty')
        required_cols = ['high', 'low']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        valid_pivot_types = ['major_low', 'major_high', 'swing_low',
            'swing_high', 'recent_low', 'recent_high']
        if pivot_type not in valid_pivot_types:
            print(
                f"Warning: Invalid pivot_type: {pivot_type}. Using 'major_low' as default."
                )
            pivot_type = 'major_low'
        if lookback_period <= 0:
            raise ValueError('lookback_period must be positive')
        lookback_data = data.iloc[-lookback_period:] if len(data
            ) > lookback_period else data
        if lookback_data.empty:
            print('Warning: Lookback data is empty')
            return None, None
        pivot_idx = None
        pivot_price = None
        try:
            if pivot_type == 'swing_low':
                pivot_idx = lookback_data['low'].idxmin()
                pivot_price = lookback_data.loc[pivot_idx, 'low']
            elif pivot_type == 'swing_high':
                pivot_idx = lookback_data['high'].idxmax()
                pivot_price = lookback_data.loc[pivot_idx, 'high']
            elif pivot_type == 'recent_low':
                pivot_idx = lookback_data.index[-1]
                pivot_price = lookback_data.loc[pivot_idx, 'low']
            elif pivot_type == 'recent_high':
                pivot_idx = lookback_data.index[-1]
                pivot_price = lookback_data.loc[pivot_idx, 'high']
            elif pivot_type == 'major_low':
                try:
                    swing_lows = find_swing_lows(lookback_data['low'], n=3)
                    if swing_lows.any() and not swing_lows.empty:
                        pivot_idx = lookback_data.loc[swing_lows].index[-1]
                        pivot_price = lookback_data.loc[pivot_idx, 'low']
                    else:
                        print(
                            'Warning: No swing lows found, falling back to minimum low'
                            )
                        pivot_idx = lookback_data['low'].idxmin()
                        pivot_price = lookback_data.loc[pivot_idx, 'low']
                except Exception as e:
                    print(
                        f'Warning: Swing point detection failed: {str(e)}. Falling back to minimum low'
                        )
                    pivot_idx = lookback_data['low'].idxmin()
                    pivot_price = lookback_data.loc[pivot_idx, 'low']
            elif pivot_type == 'major_high':
                try:
                    swing_highs = find_swing_highs(lookback_data['high'], n=3)
                    if swing_highs.any() and not swing_highs.empty:
                        pivot_idx = lookback_data.loc[swing_highs].index[-1]
                        pivot_price = lookback_data.loc[pivot_idx, 'high']
                    else:
                        print(
                            'Warning: No swing highs found, falling back to maximum high'
                            )
                        pivot_idx = lookback_data['high'].idxmax()
                        pivot_price = lookback_data.loc[pivot_idx, 'high']
                except Exception as e:
                    print(
                        f'Warning: Swing point detection failed: {str(e)}. Falling back to maximum high'
                        )
                    pivot_idx = lookback_data['high'].idxmax()
                    pivot_price = lookback_data.loc[pivot_idx, 'high']
            if pivot_idx is None or pivot_price is None:
                print(
                    f"Warning: Could not determine valid pivot point for type '{pivot_type}'"
                    )
                pivot_idx = lookback_data['low'].idxmin()
                pivot_price = lookback_data.loc[pivot_idx, 'low']
        except Exception as e:
            print(f'Error finding pivot point: {str(e)}')
            return None, None
        return pivot_idx, pivot_price

    @with_exception_handling
    def _determine_pivot(self, data: pd.DataFrame, pivot_type: str,
        lookback_period: int, auto_detect_pivot: bool=True,
        manual_pivot_price: Optional[float]=None, manual_pivot_time_idx:
        Optional[int]=None) ->Tuple[Optional[int], Optional[float]]:
        """
        Determine pivot point (position and price) based on settings.

        Args:
            data: DataFrame with OHLCV data
            pivot_type: Type of pivot ('major_low', 'major_high', 'manual')
            lookback_period: Number of bars to look back for finding pivot
            auto_detect_pivot: Whether to automatically detect the pivot
            manual_pivot_price: Manual pivot price (if pivot_type='manual')
            manual_pivot_time_idx: Manual pivot time position index (if pivot_type='manual')

        Returns:
            Tuple of (pivot_position, pivot_price)

        Raises:
            ValueError: If required data is missing or parameters are invalid
            IndexError: If manual_pivot_time_idx is out of range
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError('Data must be a pandas DataFrame')
        if len(data) == 0:
            raise ValueError('Data DataFrame is empty')
        required_cols = ['high', 'low']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        valid_pivot_types = ['major_low', 'major_high', 'swing_low',
            'swing_high', 'recent_low', 'recent_high', 'manual']
        if pivot_type not in valid_pivot_types:
            raise ValueError(
                f'Invalid pivot_type: {pivot_type}. Must be one of {valid_pivot_types}'
                )
        if lookback_period <= 0:
            raise ValueError('lookback_period must be positive')
        pivot_pos = None
        pivot_price = None
        try:
            if pivot_type == 'manual' and not auto_detect_pivot:
                if manual_pivot_price is None:
                    raise ValueError(
                        "manual_pivot_price must be provided when pivot_type='manual' and auto_detect_pivot=False"
                        )
                if manual_pivot_time_idx is None:
                    raise ValueError(
                        "manual_pivot_time_idx must be provided when pivot_type='manual' and auto_detect_pivot=False"
                        )
                if not isinstance(manual_pivot_time_idx, int):
                    raise ValueError('manual_pivot_time_idx must be an integer'
                        )
                if manual_pivot_time_idx < 0 or manual_pivot_time_idx >= len(
                    data):
                    raise IndexError(
                        f'manual_pivot_time_idx {manual_pivot_time_idx} is out of range (0-{len(data) - 1})'
                        )
                pivot_pos = manual_pivot_time_idx
                pivot_price = manual_pivot_price
            if pivot_pos is None and auto_detect_pivot:
                pivot_idx, pivot_price = self._find_pivot_point(data,
                    pivot_type, lookback_period)
                if pivot_idx is not None:
                    try:
                        pivot_pos = data.index.get_loc(pivot_idx)
                    except KeyError:
                        print(
                            f'Warning: Pivot index {pivot_idx} not found in data index'
                            )
                        return None, None
        except Exception as e:
            print(f'Error determining pivot point: {str(e)}')
            return None, None
        if pivot_pos is None or pivot_price is None:
            print(
                f"Warning: Could not determine valid pivot point for type '{pivot_type}'"
                )
        return pivot_pos, pivot_price
