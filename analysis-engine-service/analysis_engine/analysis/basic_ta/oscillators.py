"""
Oscillator Indicators Module

This module provides implementations for various oscillator-based technical indicators:
- CCI (Commodity Channel Index)
- Williams %R
- And others

These indicators help identify overbought and oversold conditions, as well as
potential trend reversals.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, List
from analysis_engine.analysis.basic_ta.base import BaseIndicator
from analysis_engine.utils.validation import validate_dataframe
logger = logging.getLogger(__name__)


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class CommodityChannelIndex(BaseIndicator):
    """
    Commodity Channel Index (CCI) indicator.
    
    The CCI measures the current price level relative to an average price level over a given period.
    It can be used to identify overbought and oversold levels, typically above +100 and below -100.
    
    Formula:
        CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
        
    where:
        TP = Typical Price = (High + Low + Close) / 3
        SMA(TP) = Simple Moving Average of Typical Price
        Mean Deviation = Mean absolute deviation of TP from its SMA
    """

    def __init__(self, period: int=20, constant: float=0.015, high_column:
        str='high', low_column: str='low', close_column: str='close', **kwargs
        ):
        """
        Initialize the CCI indicator.
        
        Args:
            period: Lookback period for calculating CCI
            constant: Scaling constant (typically 0.015)
            high_column: Column name for high prices
            low_column: Column name for low prices
            close_column: Column name for close prices
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.period = period
        self.constant = constant
        self.high_column = high_column
        self.low_column = low_column
        self.close_column = close_column
        self.output_column = f'cci_{period}'

    def calculate(self, df: pd.DataFrame) ->pd.DataFrame:
        """
        Calculate the CCI indicator for the given data.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with CCI values
        """
        validate_dataframe(df, required_columns=[self.high_column, self.
            low_column, self.close_column])
        result_df = df.copy()
        typical_price = (result_df[self.high_column] + result_df[self.
            low_column] + result_df[self.close_column]) / 3
        tp_sma = typical_price.rolling(window=self.period).mean()
        mean_deviation = np.zeros(len(typical_price))
        for i in range(self.period - 1, len(typical_price)):
            mean_deviation[i] = np.mean(np.abs(typical_price[i - self.
                period + 1:i + 1] - tp_sma[i]))
        cci = (typical_price - tp_sma) / (self.constant * mean_deviation)
        result_df[self.output_column] = cci
        return result_df

    def initialize_incremental(self) ->Dict[str, Any]:
        """Initialize state for incremental calculation."""
        return {'typical_prices': [], 'last_cci': None}

    @with_resilience('update_incremental')
    def update_incremental(self, state: Dict[str, Any], new_data: Dict[str,
        Any]) ->Dict[str, Any]:
        """Update state with new data point."""
        high = new_data.get(self.high_column, 0)
        low = new_data.get(self.low_column, 0)
        close = new_data.get(self.close_column, 0)
        typical_price = (high + low + close) / 3
        state['typical_prices'].append(typical_price)
        if len(state['typical_prices']) > self.period:
            state['typical_prices'] = state['typical_prices'][-self.period:]
        if len(state['typical_prices']) == self.period:
            tp_sma = np.mean(state['typical_prices'])
            mean_deviation = np.mean([abs(tp - tp_sma) for tp in state[
                'typical_prices']])
            if mean_deviation != 0:
                state['last_cci'] = (typical_price - tp_sma) / (self.
                    constant * mean_deviation)
            else:
                state['last_cci'] = 0
        return state


class WilliamsR(BaseIndicator):
    """
    Williams %R indicator.
    
    Williams %R is a momentum indicator that measures overbought and oversold levels,
    similar to a stochastic oscillator. It reflects the level of the close relative to the highest
    high for the lookback period.
    
    Formula:
        Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
        
    Values range from 0 to -100, with readings from 0 to -20 considered overbought
    and -80 to -100 considered oversold.
    """

    def __init__(self, period: int=14, high_column: str='high', low_column:
        str='low', close_column: str='close', **kwargs):
        """
        Initialize Williams %R indicator.
        
        Args:
            period: Lookback period
            high_column: Column name for high prices
            low_column: Column name for low prices
            close_column: Column name for close prices
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.period = period
        self.high_column = high_column
        self.low_column = low_column
        self.close_column = close_column
        self.output_column = f'williams_r_{period}'

    def calculate(self, df: pd.DataFrame) ->pd.DataFrame:
        """
        Calculate Williams %R for the given data.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with Williams %R values
        """
        validate_dataframe(df, required_columns=[self.high_column, self.
            low_column, self.close_column])
        result_df = df.copy()
        highest_high = result_df[self.high_column].rolling(window=self.period
            ).max()
        lowest_low = result_df[self.low_column].rolling(window=self.period
            ).min()
        williams_r = (highest_high - result_df[self.close_column]) / (
            highest_high - lowest_low) * -100
        result_df[self.output_column] = williams_r
        return result_df

    def initialize_incremental(self) ->Dict[str, Any]:
        """Initialize state for incremental calculation."""
        return {'highs': [], 'lows': [], 'last_williams_r': None}

    @with_resilience('update_incremental')
    def update_incremental(self, state: Dict[str, Any], new_data: Dict[str,
        Any]) ->Dict[str, Any]:
        """Update state with new data point."""
        high = new_data.get(self.high_column, 0)
        low = new_data.get(self.low_column, 0)
        close = new_data.get(self.close_column, 0)
        state['highs'].append(high)
        state['lows'].append(low)
        if len(state['highs']) > self.period:
            state['highs'] = state['highs'][-self.period:]
            state['lows'] = state['lows'][-self.period:]
        if len(state['highs']) == self.period:
            highest_high = max(state['highs'])
            lowest_low = min(state['lows'])
            if highest_high != lowest_low:
                state['last_williams_r'] = (highest_high - close) / (
                    highest_high - lowest_low) * -100
            else:
                state['last_williams_r'] = -50
        return state
