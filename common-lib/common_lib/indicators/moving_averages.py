"""
Moving Averages Module.

This module provides implementations of various moving average indicators.
It is designed to be used across multiple services to ensure consistent indicator implementation.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from common_lib.indicators.base_indicator import BaseIndicator


class SimpleMovingAverage(BaseIndicator):
    """
    Simple Moving Average (SMA) indicator.

    This indicator calculates the arithmetic mean of a given set of prices
    over a specified period.
    """

    category = "moving_average"
    name = "SimpleMovingAverage"
    default_params = {"window": 14, "column": "close"}
    required_params = {"window": int, "column": str}

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Simple Moving Average indicator.

        Args:
            params: Dictionary of parameters for the indicator calculation.
                   If None, default_params will be used.
        """
        super().__init__(params)
        self.output_column = f"sma_{self.params['window']}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate SMA for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with SMA values
        """
        self.validate_input(data, [self.params["column"]])

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Calculate SMA
        result[self.output_column] = result[self.params["column"]].rolling(
            window=self.params["window"]
        ).mean()

        return result


class ExponentialMovingAverage(BaseIndicator):
    """
    Exponential Moving Average (EMA) indicator.

    This indicator gives more weight to recent prices while still considering
    older prices with an exponentially decreasing weight.
    """

    category = "moving_average"
    name = "ExponentialMovingAverage"
    default_params = {"window": 14, "column": "close"}
    required_params = {"window": int, "column": str}

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Exponential Moving Average indicator.

        Args:
            params: Dictionary of parameters for the indicator calculation.
                   If None, default_params will be used.
        """
        super().__init__(params)
        self.output_column = f"ema_{self.params['window']}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EMA for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with EMA values
        """
        self.validate_input(data, [self.params["column"]])

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Calculate EMA - using pandas built-in EMA function
        result[self.output_column] = result[self.params["column"]].ewm(
            span=self.params["window"], adjust=False
        ).mean()

        return result


class WeightedMovingAverage(BaseIndicator):
    """
    Weighted Moving Average (WMA) indicator.

    This indicator assigns a greater weight to more recent data points
    and less weight to data points in the distant past.
    """

    category = "moving_average"
    name = "WeightedMovingAverage"
    default_params = {"window": 14, "column": "close"}
    required_params = {"window": int, "column": str}

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Weighted Moving Average indicator.

        Args:
            params: Dictionary of parameters for the indicator calculation.
                   If None, default_params will be used.
        """
        super().__init__(params)
        self.output_column = f"wma_{self.params['window']}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate WMA for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with WMA values
        """
        self.validate_input(data, [self.params["column"]])

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Create weights - higher weight for recent prices
        weights = np.arange(1, self.params["window"] + 1)
        sum_weights = np.sum(weights)

        # Calculate WMA
        result[self.output_column] = result[self.params["column"]].rolling(
            window=self.params["window"]
        ).apply(lambda x: np.sum(weights * x) / sum_weights, raw=True)

        return result