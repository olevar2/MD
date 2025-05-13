"""
Trend Indicators Module

This module provides implementations for various trend-related technical indicators:
- Parabolic SAR
- And others

These indicators help identify the direction of market trends and potential
reversal points.
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

class ParabolicSAR(BaseIndicator):
    """
    Parabolic Stop and Reverse (SAR) indicator.
    
    The Parabolic SAR is used to determine trend direction and potential reversal points.
    It places dots on a chart that indicate potential reversals in price movement.
    
    When the price is above the dots, it's generally a bullish signal.
    When price is below the dots, it's generally a bearish signal.
    
    Implementation follows Wilder's original methodology.
    """

    def __init__(self, initial_af: float=0.02, max_af: float=0.2, af_step:
        float=0.02, high_column: str='high', low_column: str='low', **kwargs):
        """
        Initialize Parabolic SAR indicator.
        
        Args:
            initial_af: Initial acceleration factor
            max_af: Maximum acceleration factor
            af_step: Acceleration factor step
            high_column: Column name for high prices
            low_column: Column name for low prices
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.initial_af = initial_af
        self.max_af = max_af
        self.af_step = af_step
        self.high_column = high_column
        self.low_column = low_column
        self.output_column = 'psar'

    def calculate(self, df: pd.DataFrame) ->pd.DataFrame:
        """
        Calculate Parabolic SAR for the given data.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with Parabolic SAR values
        """
        validate_dataframe(df, required_columns=[self.high_column, self.
            low_column])
        result_df = df.copy()
        high = result_df[self.high_column].values
        low = result_df[self.low_column].values
        psar = np.zeros(len(high))
        psar[:] = np.nan
        bull = True
        af = self.initial_af
        extreme_point = high[0]
        psar[0] = low[0]
        for i in range(1, len(high)):
            psar_prev = psar[i - 1]
            if bull:
                psar[i] = psar_prev + af * (extreme_point - psar_prev)
                psar[i] = min(psar[i], low[i - 1], low[i - 2] if i >= 2 else
                    low[i - 1])
                if psar[i] > low[i]:
                    bull = False
                    psar[i] = extreme_point
                    extreme_point = low[i]
                    af = self.initial_af
                elif high[i] > extreme_point:
                    extreme_point = high[i]
                    af = min(af + self.af_step, self.max_af)
            else:
                psar[i] = psar_prev + af * (extreme_point - psar_prev)
                psar[i] = max(psar[i], high[i - 1], high[i - 2] if i >= 2 else
                    high[i - 1])
                if psar[i] < high[i]:
                    bull = True
                    psar[i] = extreme_point
                    extreme_point = high[i]
                    af = self.initial_af
                elif low[i] < extreme_point:
                    extreme_point = low[i]
                    af = min(af + self.af_step, self.max_af)
        result_df[self.output_column] = psar
        result_df['psar_trend'] = np.where(high > psar, 1, -1)
        return result_df

    def initialize_incremental(self) ->Dict[str, Any]:
        """Initialize state for incremental calculation."""
        return {'bull': True, 'af': self.initial_af, 'extreme_point': 0,
            'previous_psar': 0, 'previous_high': 0, 'previous_low': 0,
            'first_run': True}

    @with_resilience('update_incremental')
    def update_incremental(self, state: Dict[str, Any], new_data: Dict[str,
        Any]) ->Dict[str, Any]:
        """Update state with new data point."""
        current_high = new_data.get(self.high_column, 0)
        current_low = new_data.get(self.low_column, 0)
        if state['first_run']:
            state['extreme_point'] = current_high
            state['previous_psar'] = current_low
            state['previous_high'] = current_high
            state['previous_low'] = current_low
            state['first_run'] = False
            return state
        bull = state['bull']
        af = state['af']
        extreme_point = state['extreme_point']
        psar_prev = state['previous_psar']
        if bull:
            psar = psar_prev + af * (extreme_point - psar_prev)
            psar = min(psar, state['previous_low'])
            if psar > current_low:
                bull = False
                psar = extreme_point
                extreme_point = current_low
                af = self.initial_af
            elif current_high > extreme_point:
                extreme_point = current_high
                af = min(af + self.af_step, self.max_af)
        else:
            psar = psar_prev + af * (extreme_point - psar_prev)
            psar = max(psar, state['previous_high'])
            if psar < current_high:
                bull = True
                psar = extreme_point
                extreme_point = current_high
                af = self.initial_af
            elif current_low < extreme_point:
                extreme_point = current_low
                af = min(af + self.af_step, self.max_af)
        state['bull'] = bull
        state['af'] = af
        state['extreme_point'] = extreme_point
        state['previous_psar'] = psar
        state['previous_high'] = current_high
        state['previous_low'] = current_low
        state['last_psar'] = psar
        state['last_psar_trend'] = 1 if bull else -1
        return state
