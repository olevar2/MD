"""
Pivot Point Analysis Module

This module provides various pivot point calculation methods including:
- Standard (Classic) Pivot Points
- Fibonacci Pivot Points
- Camarilla Pivot Points
- Woodie's Pivot Points
- DeMark's Pivot Points
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import math
from enum import Enum
from analysis_engine.analysis.advanced_ta.base import AdvancedAnalysisBase


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class PivotPointMethod(Enum):
    """Different pivot point calculation methods"""
    STANDARD = 'standard'
    FIBONACCI = 'fibonacci'
    CAMARILLA = 'camarilla'
    WOODIE = 'woodie'
    DEMARK = 'demark'


class PivotPoints(AdvancedAnalysisBase):
    """
    Pivot Points Analysis
    
    Calculates various types of pivot points for support and resistance levels.
    """

    def __init__(self, name: str='PivotPoints', parameters: Dict[str, Any]=None
        ):
        """
        Initialize Pivot Points analyzer
        
        Args:
            name: Name of the analyzer
            parameters: Dictionary of parameters for analysis
        """
        default_params = {'method': 'standard', 'timeframe': 'D1',
            'price_source': 'hlc', 'levels': 3}
        if parameters:
            default_params.update(parameters)
        super().__init__(name, default_params)

    @with_exception_handling
    def calculate(self, df: pd.DataFrame) ->pd.DataFrame:
        """
        Calculate pivot point levels
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with pivot point levels
        """
        result_df = df.copy()
        required_cols = ['high', 'low', 'close']
        if self.parameters['price_source'] == 'ohlc':
            required_cols.append('open')
        for col in required_cols:
            if col not in result_df.columns:
                raise ValueError(f'Required column {col} not found in data')
        try:
            pivot_method = PivotPointMethod(self.parameters['method'].lower())
        except ValueError:
            pivot_method = PivotPointMethod.STANDARD
        timeframe = self.parameters['timeframe'].upper()
        if timeframe == 'D1':
            high = result_df['high'].iloc[-2] if len(result_df
                ) > 1 else result_df['high'].iloc[-1]
            low = result_df['low'].iloc[-2] if len(result_df
                ) > 1 else result_df['low'].iloc[-1]
            close = result_df['close'].iloc[-2] if len(result_df
                ) > 1 else result_df['close'].iloc[-1]
            open_price = result_df['open'].iloc[-2
                ] if 'open' in result_df.columns and len(result_df
                ) > 1 else None
        elif timeframe == 'W1':
            recent_data = result_df.tail(8)
            high = recent_data['high'].max()
            low = recent_data['low'].min()
            close = recent_data['close'].iloc[-1]
            open_price = recent_data['open'].iloc[0
                ] if 'open' in recent_data.columns else None
        elif timeframe == 'M1':
            recent_data = result_df.tail(31)
            high = recent_data['high'].max()
            low = recent_data['low'].min()
            close = recent_data['close'].iloc[-1]
            open_price = recent_data['open'].iloc[0
                ] if 'open' in recent_data.columns else None
        else:
            high = result_df['high'].iloc[-2] if len(result_df
                ) > 1 else result_df['high'].iloc[-1]
            low = result_df['low'].iloc[-2] if len(result_df
                ) > 1 else result_df['low'].iloc[-1]
            close = result_df['close'].iloc[-2] if len(result_df
                ) > 1 else result_df['close'].iloc[-1]
            open_price = result_df['open'].iloc[-2
                ] if 'open' in result_df.columns and len(result_df
                ) > 1 else None
        levels = self.parameters['levels']
        if pivot_method == PivotPointMethod.STANDARD:
            pivot_levels = self._calculate_standard_pivots(high, low, close,
                levels)
        elif pivot_method == PivotPointMethod.FIBONACCI:
            pivot_levels = self._calculate_fibonacci_pivots(high, low,
                close, levels)
        elif pivot_method == PivotPointMethod.CAMARILLA:
            pivot_levels = self._calculate_camarilla_pivots(high, low,
                close, levels)
        elif pivot_method == PivotPointMethod.WOODIE:
            pivot_levels = self._calculate_woodie_pivots(high, low, close,
                open_price, levels)
        elif pivot_method == PivotPointMethod.DEMARK:
            pivot_levels = self._calculate_demark_pivots(high, low, close,
                open_price, levels)
        else:
            pivot_levels = self._calculate_standard_pivots(high, low, close,
                levels)
        prefix = f'pivot_{pivot_method.value}'
        pp_column = f'{prefix}_pp'
        result_df[pp_column] = pivot_levels['PP']
        for i in range(1, levels + 1):
            r_column = f'{prefix}_r{i}'
            if f'R{i}' in pivot_levels:
                result_df[r_column] = pivot_levels[f'R{i}']
        for i in range(1, levels + 1):
            s_column = f'{prefix}_s{i}'
            if f'S{i}' in pivot_levels:
                result_df[s_column] = pivot_levels[f'S{i}']
        return result_df

    def _calculate_standard_pivots(self, high: float, low: float, close:
        float, levels: int) ->Dict[str, float]:
        """
        Calculate standard (classic) pivot points
        
        Args:
            high: High price
            low: Low price
            close: Close price
            levels: Number of support/resistance levels
            
        Returns:
            Dictionary of pivot point levels
        """
        pp = (high + low + close) / 3
        result = {'PP': pp}
        result['R1'] = 2 * pp - low
        result['R2'] = pp + (high - low)
        result['R3'] = high + 2 * (pp - low)
        result['S1'] = 2 * pp - high
        result['S2'] = pp - (high - low)
        result['S3'] = low - 2 * (high - pp)
        if levels > 3:
            result['R4'] = result['R3'] + (high - low)
            result['S4'] = result['S3'] - (high - low)
            if levels > 4:
                result['R5'] = result['R4'] + (high - low)
                result['S5'] = result['S4'] - (high - low)
        return result

    def _calculate_fibonacci_pivots(self, high: float, low: float, close:
        float, levels: int) ->Dict[str, float]:
        """
        Calculate Fibonacci pivot points
        
        Args:
            high: High price
            low: Low price
            close: Close price
            levels: Number of support/resistance levels
            
        Returns:
            Dictionary of pivot point levels
        """
        pp = (high + low + close) / 3
        result = {'PP': pp}
        result['R1'] = pp + 0.382 * (high - low)
        result['R2'] = pp + 0.618 * (high - low)
        result['R3'] = pp + 1.0 * (high - low)
        result['S1'] = pp - 0.382 * (high - low)
        result['S2'] = pp - 0.618 * (high - low)
        result['S3'] = pp - 1.0 * (high - low)
        if levels > 3:
            result['R4'] = pp + 1.618 * (high - low)
            result['S4'] = pp - 1.618 * (high - low)
            if levels > 4:
                result['R5'] = pp + 2.618 * (high - low)
                result['S5'] = pp - 2.618 * (high - low)
        return result

    def _calculate_camarilla_pivots(self, high: float, low: float, close:
        float, levels: int) ->Dict[str, float]:
        """
        Calculate Camarilla pivot points
        
        Args:
            high: High price
            low: Low price
            close: Close price
            levels: Number of support/resistance levels
            
        Returns:
            Dictionary of pivot point levels
        """
        pp = (high + low + close) / 3
        result = {'PP': pp}
        range_hl = high - low
        result['R1'] = close + range_hl * 1.1 / 12
        result['R2'] = close + range_hl * 1.1 / 6
        result['R3'] = close + range_hl * 1.1 / 4
        result['R4'] = close + range_hl * 1.1 / 2
        result['S1'] = close - range_hl * 1.1 / 12
        result['S2'] = close - range_hl * 1.1 / 6
        result['S3'] = close - range_hl * 1.1 / 4
        result['S4'] = close - range_hl * 1.1 / 2
        if levels > 4:
            result['R5'] = result['R4'] + range_hl * 1.1 / 4
            result['S5'] = result['S4'] - range_hl * 1.1 / 4
        return result

    def _calculate_woodie_pivots(self, high: float, low: float, close:
        float, open_price: Optional[float], levels: int) ->Dict[str, float]:
        """
        Calculate Woodie's pivot points
        
        Args:
            high: High price
            low: Low price
            close: Close price
            open_price: Open price (if available)
            levels: Number of support/resistance levels
            
        Returns:
            Dictionary of pivot point levels
        """
        open_curr = open_price if open_price is not None else close
        pp = (high + low + 2 * close) / 4
        result = {'PP': pp}
        result['R1'] = 2 * pp - low
        result['R2'] = pp + high - low
        result['S1'] = 2 * pp - high
        result['S2'] = pp - high + low
        if levels > 2:
            result['R3'] = high + 2 * (pp - low)
            result['S3'] = low - 2 * (high - pp)
            if levels > 3:
                result['R4'] = result['R3'] + (high - low)
                result['S4'] = result['S3'] - (high - low)
        return result

    def _calculate_demark_pivots(self, high: float, low: float, close:
        float, open_price: Optional[float], levels: int) ->Dict[str, float]:
        """
        Calculate DeMark's pivot points
        
        Args:
            high: High price
            low: Low price
            close: Close price
            open_price: Open price (if available)
            levels: Number of support/resistance levels
            
        Returns:
            Dictionary of pivot point levels
        """
        if close > open_price if open_price is not None else 0:
            x = 2 * high + low + close
        elif close < open_price if open_price is not None else 0:
            x = high + 2 * low + close
        else:
            x = high + low + 2 * close
        pp = x / 4
        result = {'PP': pp}
        result['R1'] = x / 2 - low
        result['S1'] = x / 2 - high
        if levels > 1:
            result['R2'] = pp + (result['R1'] - pp)
            result['S2'] = pp - (pp - result['S1'])
            if levels > 2:
                result['R3'] = result['R2'] + (result['R2'] - pp)
                result['S3'] = result['S2'] - (pp - result['S2'])
        return result

    @classmethod
    def get_info(cls) ->Dict[str, Any]:
        """Get indicator information"""
        return {'name': 'Pivot Points', 'description':
            'Calculates various types of pivot points for support and resistance'
            , 'category': 'support_resistance', 'parameters': [{'name':
            'method', 'description': 'Pivot point calculation method',
            'type': 'str', 'default': 'standard', 'options': ['standard',
            'fibonacci', 'camarilla', 'woodie', 'demark']}, {'name':
            'timeframe', 'description':
            'Timeframe to use for pivot calculations', 'type': 'str',
            'default': 'D1', 'options': ['D1', 'W1', 'M1']}, {'name':
            'price_source', 'description':
            'Price data to use for calculations', 'type': 'str', 'default':
            'hlc', 'options': ['hlc', 'ohlc']}, {'name': 'levels',
            'description':
            'Number of support/resistance levels to calculate', 'type':
            'int', 'default': 3, 'min': 1, 'max': 5}]}
