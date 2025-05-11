"""
Moving Average Convergence Divergence (MACD) Indicator

This module provides the implementation of the Moving Average Convergence Divergence (MACD) indicator.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple

from common_lib.indicators.base_indicator import BaseIndicator


class MACD(BaseIndicator):
    """
    Moving Average Convergence Divergence (MACD) Indicator.
    
    The MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of a security's price.
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        price_column: str = "close",
        name: str = "macd",
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the MACD indicator.
        
        Args:
            fast_period: Period for the fast EMA
            slow_period: Period for the slow EMA
            signal_period: Period for the signal line
            price_column: Column to use for the price data
            name: Name of the indicator
            params: Additional parameters for the indicator
        """
        params = params or {}
        params.update({
            "fast_period": fast_period,
            "slow_period": slow_period,
            "signal_period": signal_period,
            "price_column": price_column
        })
        
        output_columns = [
            f"{name}_line",
            f"{name}_signal",
            f"{name}_histogram"
        ]
        
        super().__init__(
            name=name,
            params=params,
            input_columns=[price_column],
            output_columns=output_columns
        )
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the MACD values.
        
        Args:
            data: Input data for the calculation
            
        Returns:
            DataFrame containing the MACD values
        """
        # Validate input data
        if not self.validate_input(data):
            return pd.DataFrame()
        
        # Get parameters
        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]
        signal_period = self.params["signal_period"]
        price_column = self.params["price_column"]
        
        # Calculate EMAs
        fast_ema = data[price_column].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data[price_column].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Create result DataFrame
        result = pd.DataFrame(index=data.index)
        result[f"{self.name}_line"] = macd_line
        result[f"{self.name}_signal"] = signal_line
        result[f"{self.name}_histogram"] = histogram
        
        return result
    
    def calculate_incremental(self, data: pd.DataFrame, previous_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the MACD values incrementally.
        
        This method is optimized for incremental updates, where only a small amount of new data
        is added to the existing data.
        
        Args:
            data: New data for the calculation
            previous_data: Previous data with calculated MACD values
            
        Returns:
            DataFrame containing the MACD values for the new data
        """
        # Validate input data
        if not self.validate_input(data):
            return pd.DataFrame()
        
        # Get parameters
        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]
        signal_period = self.params["signal_period"]
        price_column = self.params["price_column"]
        
        # Check if previous data has the required columns
        required_columns = [
            f"{self.name}_line",
            f"{self.name}_signal",
            f"{self.name}_histogram"
        ]
        
        if (previous_data is None or previous_data.empty or
                not all(col in previous_data.columns for col in required_columns)):
            return self.calculate(data)
        
        # Combine previous and new data
        combined_data = pd.concat([previous_data, data]).sort_index()
        
        # Remove duplicates
        combined_data = combined_data[~combined_data.index.duplicated(keep="last")]
        
        # Calculate EMAs
        fast_ema = combined_data[price_column].ewm(span=fast_period, adjust=False).mean()
        slow_ema = combined_data[price_column].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Create result DataFrame
        result = pd.DataFrame(index=combined_data.index)
        result[f"{self.name}_line"] = macd_line
        result[f"{self.name}_signal"] = signal_line
        result[f"{self.name}_histogram"] = histogram
        
        # Return only the new data
        return result.loc[data.index]