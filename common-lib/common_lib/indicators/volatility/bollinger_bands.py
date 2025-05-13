"""
Bollinger Bands Indicator

This module provides the implementation of the Bollinger Bands indicator.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple

from common_lib.indicators.base_indicator import BaseIndicator


class BollingerBands(BaseIndicator):
    """
    Bollinger Bands Indicator.
    
    Bollinger Bands consist of a middle band (simple moving average) and two outer bands
    that are standard deviations away from the middle band.
    """
    
    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        price_column: str = "close",
        name: str = "bollinger_bands",
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Bollinger Bands indicator.
        
        Args:
            period: Period for the moving average
            std_dev: Number of standard deviations for the outer bands
            price_column: Column to use for the price data
            name: Name of the indicator
            params: Additional parameters for the indicator
        """
        params = params or {}
        params.update({
            "period": period,
            "std_dev": std_dev,
            "price_column": price_column
        })
        
        output_columns = [
            f"{name}_upper",
            f"{name}_middle",
            f"{name}_lower",
            f"{name}_width",
            f"{name}_percent_b"
        ]
        
        super().__init__(
            name=name,
            params=params,
            input_columns=[price_column],
            output_columns=output_columns
        )
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Bollinger Bands values.
        
        Args:
            data: Input data for the calculation
            
        Returns:
            DataFrame containing the Bollinger Bands values
        """
        # Validate input data
        if not self.validate_input(data):
            return pd.DataFrame()
        
        # Get parameters
        period = self.params["period"]
        std_dev = self.params["std_dev"]
        price_column = self.params["price_column"]
        
        # Calculate middle band (SMA)
        middle_band = data[price_column].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = data[price_column].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        # Calculate bandwidth
        bandwidth = (upper_band - lower_band) / middle_band
        
        # Calculate %B
        percent_b = (data[price_column] - lower_band) / (upper_band - lower_band)
        
        # Create result DataFrame
        result = pd.DataFrame(index=data.index)
        result[f"{self.name}_upper"] = upper_band
        result[f"{self.name}_middle"] = middle_band
        result[f"{self.name}_lower"] = lower_band
        result[f"{self.name}_width"] = bandwidth
        result[f"{self.name}_percent_b"] = percent_b
        
        return result
    
    def calculate_incremental(self, data: pd.DataFrame, previous_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Bollinger Bands values incrementally.
        
        This method is optimized for incremental updates, where only a small amount of new data
        is added to the existing data.
        
        Args:
            data: New data for the calculation
            previous_data: Previous data with calculated Bollinger Bands values
            
        Returns:
            DataFrame containing the Bollinger Bands values for the new data
        """
        # Validate input data
        if not self.validate_input(data):
            return pd.DataFrame()
        
        # Get parameters
        period = self.params["period"]
        std_dev = self.params["std_dev"]
        price_column = self.params["price_column"]
        
        # Check if previous data has the required columns
        required_columns = [
            f"{self.name}_upper",
            f"{self.name}_middle",
            f"{self.name}_lower",
            f"{self.name}_width",
            f"{self.name}_percent_b"
        ]
        
        if (previous_data is None or previous_data.empty or
                not all(col in previous_data.columns for col in required_columns)):
            return self.calculate(data)
        
        # Combine previous and new data
        combined_data = pd.concat([previous_data, data]).sort_index()
        
        # Remove duplicates
        combined_data = combined_data[~combined_data.index.duplicated(keep="last")]
        
        # Calculate middle band (SMA)
        middle_band = combined_data[price_column].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = combined_data[price_column].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        # Calculate bandwidth
        bandwidth = (upper_band - lower_band) / middle_band
        
        # Calculate %B
        percent_b = (combined_data[price_column] - lower_band) / (upper_band - lower_band)
        
        # Create result DataFrame
        result = pd.DataFrame(index=combined_data.index)
        result[f"{self.name}_upper"] = upper_band
        result[f"{self.name}_middle"] = middle_band
        result[f"{self.name}_lower"] = lower_band
        result[f"{self.name}_width"] = bandwidth
        result[f"{self.name}_percent_b"] = percent_b
        
        # Return only the new data
        return result.loc[data.index]