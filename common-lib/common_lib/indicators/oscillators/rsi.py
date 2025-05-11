"""
Relative Strength Index (RSI) Indicator

This module provides the implementation of the Relative Strength Index (RSI) indicator.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple

from common_lib.indicators.base_indicator import BaseIndicator


class RSI(BaseIndicator):
    """
    Relative Strength Index (RSI) Indicator.
    
    The RSI is a momentum oscillator that measures the speed and change of price movements.
    It oscillates between 0 and 100 and is typically used to identify overbought or oversold
    conditions in a market.
    """
    
    def __init__(
        self,
        period: int = 14,
        price_column: str = "close",
        name: str = "rsi",
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the RSI indicator.
        
        Args:
            period: Period for the RSI calculation
            price_column: Column to use for the price data
            name: Name of the indicator
            params: Additional parameters for the indicator
        """
        params = params or {}
        params.update({
            "period": period,
            "price_column": price_column
        })
        
        super().__init__(
            name=name,
            params=params,
            input_columns=[price_column],
            output_columns=[name]
        )
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the RSI values.
        
        Args:
            data: Input data for the calculation
            
        Returns:
            DataFrame containing the RSI values
        """
        # Validate input data
        if not self.validate_input(data):
            return pd.DataFrame()
        
        # Get parameters
        period = self.params["period"]
        price_column = self.params["price_column"]
        
        # Calculate price changes
        delta = data[price_column].diff()
        
        # Calculate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Create result DataFrame
        result = pd.DataFrame(index=data.index)
        result[self.name] = rsi
        
        return result
    
    def calculate_incremental(self, data: pd.DataFrame, previous_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the RSI values incrementally.
        
        This method is optimized for incremental updates, where only a small amount of new data
        is added to the existing data.
        
        Args:
            data: New data for the calculation
            previous_data: Previous data with calculated RSI values
            
        Returns:
            DataFrame containing the RSI values for the new data
        """
        # Validate input data
        if not self.validate_input(data):
            return pd.DataFrame()
        
        # Get parameters
        period = self.params["period"]
        price_column = self.params["price_column"]
        
        # If there's no previous data or it doesn't have the RSI column, calculate from scratch
        if previous_data is None or previous_data.empty or self.name not in previous_data.columns:
            return self.calculate(data)
        
        # Combine previous and new data
        combined_data = pd.concat([previous_data, data]).sort_index()
        
        # Remove duplicates
        combined_data = combined_data[~combined_data.index.duplicated(keep="last")]
        
        # Calculate price changes
        delta = combined_data[price_column].diff()
        
        # Calculate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Create result DataFrame
        result = pd.DataFrame(index=combined_data.index)
        result[self.name] = rsi
        
        # Return only the new data
        return result.loc[data.index]