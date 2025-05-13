"""
Statistical and Regression Indicators Module.

This module provides implementations of various statistical and regression-based indicators.
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats # Import scipy for linear regression
from feature_store_service.indicators.base_indicator import BaseIndicator
from feature_store_service.indicators.moving_averages import (
    SimpleMovingAverage, ExponentialMovingAverage, WeightedMovingAverage
)


class StandardDeviationIndicator(BaseIndicator):
    """
    Standard Deviation Indicator with probability distributions.

    This indicator measures the dispersion of price values relative to their average,
    providing insights into market volatility and probability-based analysis.
    """

    category = "statistical"
    default_params = {
        'window': 20, 'column': 'close', 'bands': [1.0, 2.0, 3.0],
        'moving_average_type': 'sma', 'include_probability': False, # Defaulting probability to False for simplicity
        'distribution': 'normal', 'confidence_levels': None
    }

    def __init__(
        self,
        window: int = 20,
        column: str = "close",
        bands: Optional[List[float]] = None,
        moving_average_type: str = "sma",
        include_probability: bool = False,
        distribution: str = "normal",
        confidence_levels: Optional[List[float]] = None,
        **kwargs
    ):
    """
      init  .
    
    Args:
        window: Description of window
        column: Description of column
        bands: Description of bands
        moving_average_type: Description of moving_average_type
        include_probability: Description of include_probability
        distribution: Description of distribution
        confidence_levels: Description of confidence_levels
        kwargs: Description of kwargs
    
    """

        self.window = window
        self.column = column
        self.bands = bands if bands is not None else [1.0, 2.0, 3.0]
        self.moving_average_type = moving_average_type.lower()
        self.include_probability = include_probability # Keep the option, but calculation not implemented here
        self.distribution = distribution # Keep the option
        self.confidence_levels = confidence_levels # Keep the option
        self.name_base = f"stddev_{column}_{window}_{moving_average_type}"
        self.ma_name = f"{moving_average_type}_{window}"
        self.std_dev_name = f"{self.name_base}_value"
        super().__init__(**kwargs)

    def _get_moving_average(self, series: pd.Series) -> pd.Series:
        """Helper to calculate the specified moving average."""
        if self.moving_average_type == 'sma':
            ma_indicator = SimpleMovingAverage(window=self.window, column=series.name)
            return ma_indicator.calculate(pd.DataFrame({series.name: series}))[ma_indicator.name]
        elif self.moving_average_type == 'ema':
            ma_indicator = ExponentialMovingAverage(window=self.window, column=series.name)
            return ma_indicator.calculate(pd.DataFrame({series.name: series}))[ma_indicator.name]
        elif self.moving_average_type == 'wma':
            ma_indicator = WeightedMovingAverage(window=self.window, column=series.name)
            return ma_indicator.calculate(pd.DataFrame({series.name: series}))[ma_indicator.name]
        else:
            raise ValueError(f"Unsupported moving average type: {self.moving_average_type}")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate.
    
    Args:
        data: Description of data
    
    Returns:
        pd.DataFrame: Description of return value
    
    """

        self.validate_input(data, [self.column])
        result = data.copy()

        # Calculate the rolling standard deviation
        std_dev = result[self.column].rolling(window=self.window).std()
        result[self.std_dev_name] = std_dev.fillna(method='bfill')

        # Calculate the central moving average
        moving_avg = self._get_moving_average(result[self.column])
        result[self.ma_name] = moving_avg.fillna(method='bfill')

        # Calculate standard deviation bands
        for band_multiplier in self.bands:
            band_multiplier_str = str(band_multiplier).replace('.', '_')
            upper_band_name = f"{self.name_base}_upper_{band_multiplier_str}"
            lower_band_name = f"{self.name_base}_lower_{band_multiplier_str}"
            result[upper_band_name] = result[self.ma_name] + (band_multiplier * result[self.std_dev_name])
            result[lower_band_name] = result[self.ma_name] - (band_multiplier * result[self.std_dev_name])

        # Placeholder for probability calculations if include_probability is True
        # This would involve using scipy.stats distributions (e.g., norm, t)
        # based on the calculated mean (moving_avg) and std_dev.
        # Example: result[f'{self.name_base}_prob_within_1std'] = stats.norm.cdf(1) - stats.norm.cdf(-1)

        return result


class LinearRegressionIndicator(BaseIndicator):
    """
    Linear Regression Indicator.

    Calculates the slope, intercept, and predicted value of a linear regression
    line fitted to the data over a rolling window.
    """

    category = "statistical"
    default_params = {'window': 14, 'column': 'close'}

    def __init__(self, window: int = 14, column: str = "close", **kwargs):
    """
      init  .
    
    Args:
        window: Description of window
        column: Description of column
        kwargs: Description of kwargs
    
    """

        self.window = window
        self.column = column
        self.name_base = f"linreg_{column}_{window}"
        self.slope_name = f"{self.name_base}_slope"
        self.intercept_name = f"{self.name_base}_intercept"
        self.predicted_name = f"{self.name_base}_predicted" # Value on the regression line at the end of the window
        super().__init__(**kwargs)

    def _calculate_rolling_regression(self, y_values):
    """
     calculate rolling regression.
    
    Args:
        y_values: Description of y_values
    
    """

        # x values are simply the sequence 0, 1, ..., window-1
        x_values = np.arange(self.window)
        # Use scipy's linregress for efficiency
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
        # Return slope, intercept, and the predicted value at the *last* point (x = window - 1)
        predicted_value = intercept + slope * (self.window - 1)
        return slope, intercept, predicted_value

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate.
    
    Args:
        data: Description of data
    
    Returns:
        pd.DataFrame: Description of return value
    
    """

        self.validate_input(data, [self.column])
        result = data.copy()

        # Apply the rolling regression calculation
        regression_results = result[self.column].rolling(window=self.window).apply(
            self._calculate_rolling_regression,
            raw=True # Pass numpy arrays for speed
        )

        # Unpack the results
        # Check if results are tuples before unpacking
        valid_results = regression_results.dropna()
        if not valid_results.empty and isinstance(valid_results.iloc[0], tuple):
            result[self.slope_name] = valid_results.apply(lambda x: x[0])
            result[self.intercept_name] = valid_results.apply(lambda x: x[1])
            result[self.predicted_name] = valid_results.apply(lambda x: x[2])
        else: # Handle cases with insufficient data or single-value results
            result[self.slope_name] = np.nan
            result[self.intercept_name] = np.nan
            result[self.predicted_name] = np.nan

        # Backfill NaNs for visualization/continuity
        result[self.slope_name] = result[self.slope_name].fillna(method='bfill')
        result[self.intercept_name] = result[self.intercept_name].fillna(method='bfill')
        result[self.predicted_name] = result[self.predicted_name].fillna(method='bfill')

        return result


class LinearRegressionChannel(BaseIndicator):
    """
    Linear Regression Channel.

    Draws parallel lines above and below a linear regression trendline,
    based on the maximum deviation of the price from the trendline within the window.
    """

    category = "statistical"
    default_params = {'window': 14, 'column': 'close'}

    def __init__(self, window: int = 14, column: str = "close", **kwargs):
    """
      init  .
    
    Args:
        window: Description of window
        column: Description of column
        kwargs: Description of kwargs
    
    """

        self.window = window
        self.column = column
        self.name_base = f"linregchannel_{column}_{window}"
        self.upper_name = f"{self.name_base}_upper"
        self.lower_name = f"{self.name_base}_lower"
        self.center_name = f"{self.name_base}_center" # Same as linreg predicted
        super().__init__(**kwargs)

    def _calculate_rolling_channel(self, y_values):
    """
     calculate rolling channel.
    
    Args:
        y_values: Description of y_values
    
    """

        x_values = np.arange(self.window)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)

        # Calculate predicted values along the regression line within the window
        predicted_values_in_window = intercept + slope * x_values

        # Calculate deviations from the regression line
        deviations = y_values - predicted_values_in_window

        # Find the maximum absolute deviation
        max_deviation = np.max(np.abs(deviations))

        # Calculate channel lines at the *end* of the window (x = window - 1)
        center_value = intercept + slope * (self.window - 1)
        upper_value = center_value + max_deviation
        lower_value = center_value - max_deviation

        return upper_value, lower_value, center_value

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate.
    
    Args:
        data: Description of data
    
    Returns:
        pd.DataFrame: Description of return value
    
    """

        self.validate_input(data, [self.column])
        result = data.copy()

        # Apply the rolling channel calculation
        channel_results = result[self.column].rolling(window=self.window).apply(
            self._calculate_rolling_channel,
            raw=True
        )

        # Unpack the results
        valid_results = channel_results.dropna()
        if not valid_results.empty and isinstance(valid_results.iloc[0], tuple):
            result[self.upper_name] = valid_results.apply(lambda x: x[0])
            result[self.lower_name] = valid_results.apply(lambda x: x[1])
            result[self.center_name] = valid_results.apply(lambda x: x[2])
        else:
            result[self.upper_name] = np.nan
            result[self.lower_name] = np.nan
            result[self.center_name] = np.nan

        # Backfill NaNs
        result[self.upper_name] = result[self.upper_name].fillna(method='bfill')
        result[self.lower_name] = result[self.lower_name].fillna(method='bfill')
        result[self.center_name] = result[self.center_name].fillna(method='bfill')

        return result


class RSquaredIndicator(BaseIndicator):
    """
    R-Squared Indicator.

    Measures the goodness of fit of a linear regression line to the data
    over a rolling window. Values range from 0 to 1.
    """

    category = "statistical"
    default_params = {'window': 14, 'column': 'close'}

    def __init__(self, window: int = 14, column: str = "close", **kwargs):
    """
      init  .
    
    Args:
        window: Description of window
        column: Description of column
        kwargs: Description of kwargs
    
    """

        self.window = window
        self.column = column
        self.name = f"rsquared_{column}_{window}"
        super().__init__(**kwargs)

    def _calculate_rolling_r_squared(self, y_values):
    """
     calculate rolling r squared.
    
    Args:
        y_values: Description of y_values
    
    """

        x_values = np.arange(self.window)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
        return r_value**2 # R-squared is the square of the correlation coefficient (r_value)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate.
    
    Args:
        data: Description of data
    
    Returns:
        pd.DataFrame: Description of return value
    
    """

        self.validate_input(data, [self.column])
        result = data.copy()

        # Apply the rolling R-squared calculation
        r_squared_results = result[self.column].rolling(window=self.window).apply(
            self._calculate_rolling_r_squared,
            raw=True
        )

        result[self.name] = r_squared_results.fillna(method='bfill')

        return result
