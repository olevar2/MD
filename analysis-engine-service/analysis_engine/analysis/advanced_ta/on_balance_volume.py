"""
On-Balance Volume (OBV) Indicator Module

This module implements the On-Balance Volume (OBV) indicator, which relates volume to price change.
OBV provides insight into how volume affects price movements, serving as a leading indicator
that can help predict price movements before they occur.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, List
from analysis_engine.analysis.advanced_ta.base import AdvancedAnalysisBase
from analysis_engine.utils.validation import validate_dataframe
logger = logging.getLogger(__name__)


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class OnBalanceVolume(AdvancedAnalysisBase):
    """
    On-Balance Volume (OBV) Indicator
    
    OBV is a momentum indicator that uses volume flow to predict changes in price.
    It adds volume on up days and subtracts volume on down days, creating a running total.
    
    The indicator helps identify:
    - Divergences between price and volume
    - Potential trend confirmations
    - Volume-based support and resistance levels
    """

    def __init__(self, ma_period: int=20, price_column: str='close',
        volume_column: str='volume', output_prefix: str='OBV', **kwargs):
        """
        Initialize On-Balance Volume indicator.
        
        Args:
            ma_period: Period for the OBV moving average
            price_column: Name of the price column to use
            volume_column: Name of the volume column to use
            output_prefix: Prefix for output column names
            **kwargs: Additional parameters
        """
        parameters = {'ma_period': ma_period, 'price_column': price_column,
            'volume_column': volume_column, 'output_prefix': output_prefix}
        super().__init__('On-Balance Volume', parameters)

    def calculate(self, df: pd.DataFrame) ->pd.DataFrame:
        """
        Calculate On-Balance Volume (OBV) for the given data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with OBV values
        """
        validate_dataframe(df, required_columns=[self.parameters[
            'price_column'], self.parameters['volume_column']])
        result_df = df.copy()
        price_col = self.parameters['price_column']
        vol_col = self.parameters['volume_column']
        ma_period = self.parameters['ma_period']
        prefix = self.parameters['output_prefix']
        result_df[f'{prefix}'] = 0.0
        if len(result_df) > 0:
            result_df[f'{prefix}'].iloc[0] = result_df[vol_col].iloc[0]
        for i in range(1, len(result_df)):
            price_change = result_df[price_col].iloc[i] - result_df[price_col
                ].iloc[i - 1]
            prev_obv = result_df[f'{prefix}'].iloc[i - 1]
            current_volume = result_df[vol_col].iloc[i]
            if price_change > 0:
                result_df.loc[result_df.index[i], f'{prefix}'
                    ] = prev_obv + current_volume
            elif price_change < 0:
                result_df.loc[result_df.index[i], f'{prefix}'
                    ] = prev_obv - current_volume
            else:
                result_df.loc[result_df.index[i], f'{prefix}'] = prev_obv
        result_df[f'{prefix}_MA'] = result_df[f'{prefix}'].rolling(window=
            ma_period).mean()
        return result_df

    def initialize_incremental(self) ->Dict[str, Any]:
        """
        Initialize state for incremental calculation
        
        Returns:
            Initial state dictionary
        """
        state = {'last_price': None, 'last_obv': 0.0, 'ma_values': [],
            'ma_period': self.parameters['ma_period']}
        return state

    @with_resilience('update_incremental')
    def update_incremental(self, state: Dict[str, Any], new_data: Dict[str,
        Any]) ->Dict[str, Any]:
        """
        Update OBV calculation with new data incrementally
        
        Args:
            state: Current calculation state
            new_data: New data point
            
        Returns:
            Updated state and OBV values
        """
        price_col = self.parameters['price_column']
        vol_col = self.parameters['volume_column']
        ma_period = state['ma_period']
        if price_col not in new_data or vol_col not in new_data:
            logger.warning(
                f'Required columns {price_col} or {vol_col} missing in new_data'
                )
            return state
        current_price = new_data[price_col]
        current_volume = new_data[vol_col]
        if state['last_price'] is None:
            current_obv = current_volume
        else:
            price_change = current_price - state['last_price']
            if price_change > 0:
                current_obv = state['last_obv'] + current_volume
            elif price_change < 0:
                current_obv = state['last_obv'] - current_volume
            else:
                current_obv = state['last_obv']
        state['last_price'] = current_price
        state['last_obv'] = current_obv
        state['ma_values'].append(current_obv)
        if len(state['ma_values']) > ma_period:
            state['ma_values'] = state['ma_values'][-ma_period:]
        if len(state['ma_values']) > 0:
            current_ma = sum(state['ma_values']) / len(state['ma_values'])
        else:
            current_ma = None
        state['current_obv'] = current_obv
        state['current_ma'] = current_ma
        return state
