"""
Statistical and Regression Indicators Module

This module provides implementations of statistical and regression-based indicators
including Standard Deviation, Linear Regression, and Linear Regression Channel
for analyzing statistical properties of price movements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats

from analysis_engine.analysis.advanced_ta.base import AdvancedAnalysisBase


class StandardDeviationAnalyzer(AdvancedAnalysisBase):
    """
    Standard Deviation Analyzer
    
    Measures the dispersion of price values relative to their average,
    providing insights into market volatility and potential price movements.
    """
    
    def __init__(
        self,
        name: str = "StandardDeviation",
        window: int = 20,
        column: str = "close",
        bands: List[float] = None,
        moving_average_type: str = "sma",
        **kwargs
    ):
        """Initialize the Standard Deviation analyzer.
        
        Args:
            name: Identifier for this analyzer
            window: Lookback period for calculations
            column: Data column to use for calculations
            bands: List of multipliers for standard deviation bands
            moving_average_type: Type of moving average to use ('sma', 'ema', 'wma')
            **kwargs: Additional parameters
        """
        super().__init__(name=name, **kwargs)
        self.window = window
        self.column = column
        self.bands = bands if bands is not None else [1.0, 2.0, 3.0]
        self.moving_average_type = moving_average_type.lower()
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Standard Deviation indicator.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with Standard Deviation results
        """
        if self.column not in data.columns:
            raise ValueError(f"Data must contain '{self.column}' column")
        
        # Calculate moving average based on type
        if self.moving_average_type == "sma":
            ma = data[self.column].rolling(window=self.window).mean()
        elif self.moving_average_type == "ema":
            ma = data[self.column].ewm(span=self.window, adjust=False).mean()
        elif self.moving_average_type == "wma":
            weights = np.arange(1, self.window + 1)
            ma = data[self.column].rolling(window=self.window).apply(
                lambda x: np.sum(weights * x) / weights.sum(), raw=True
            )
        else:
            raise ValueError(f"Unsupported moving average type: {self.moving_average_type}")
            
        # Calculate standard deviation
        std = data[self.column].rolling(window=self.window).std()
        
        results = {
            "std": std,
            f"{self.moving_average_type}": ma
        }
        
        # Calculate standard deviation bands
        for band in self.bands:
            results[f"std_band_{band}_upper"] = ma + (band * std)
            results[f"std_band_{band}_lower"] = ma - (band * std)
        
        # Add volatility prediction
        if len(data) > self.window * 3:
            # Calculate historical standard deviation of the standard deviation
            std_of_std = std.rolling(window=self.window * 3).std()
            avg_of_std = std.rolling(window=self.window * 3).mean()
            
            # Calculate z-score of current std relative to historical distribution
            vol_pred = (std - avg_of_std) / std_of_std
            
            results["vol_pred"] = vol_pred
            
        self.results = results
        return results


class LinearRegressionAnalyzer(AdvancedAnalysisBase):
    """
    Linear Regression Analyzer
    
    Applies linear regression to price data to generate a regression line,
    slope, and R-squared value, useful for trend analysis.
    """
    
    def __init__(
        self,
        name: str = "LinearRegression",
        window: int = 20,
        column: str = "close",
        forecast_periods: int = 0,
        **kwargs
    ):
        """Initialize the Linear Regression analyzer.
        
        Args:
            name: Identifier for this analyzer
            window: Lookback period for regression calculations
            column: Data column to use for calculations
            forecast_periods: Number of periods to forecast (0 for no forecast)
            **kwargs: Additional parameters
        """
        super().__init__(name=name, **kwargs)
        self.window = window
        self.column = column
        self.forecast_periods = forecast_periods
        
    def _calculate_regression(self, y: pd.Series) -> Dict[str, Any]:
        """Calculate linear regression for a given series."""
        n = len(y)
        if n <= 1:
            return {
                'slope': np.nan,
                'intercept': np.nan,
                'r2': np.nan,
                'angle': np.nan,
                'line': np.full(n, np.nan)
            }
            
        x = np.arange(n)
        
        # Calculate linear regression using scipy.stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate regression line
        line = intercept + slope * x
        
        # Calculate angle in degrees
        angle = np.degrees(np.arctan(slope))
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r2': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
            'angle': angle,
            'line': line
        }
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Linear Regression indicator.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with Linear Regression results
        """
        if self.column not in data.columns:
            raise ValueError(f"Data must contain '{self.column}' column")
        
        n = len(data)
        results = {
            'linreg_line': np.full(n, np.nan),
            'linreg_slope': np.full(n, np.nan),
            'linreg_r2': np.full(n, np.nan),
            'linreg_angle': np.full(n, np.nan),
        }
        
        if self.forecast_periods > 0:
            results['linreg_forecast'] = np.full(n, np.nan)
        
        # Apply rolling regression
        for i in range(self.window - 1, n):
            window_slice = data[self.column].iloc[i - self.window + 1:i + 1].values
            reg_result = self._calculate_regression(window_slice)
            
            results['linreg_line'][i] = reg_result['line'][-1]
            results['linreg_slope'][i] = reg_result['slope']
            results['linreg_r2'][i] = reg_result['r2']
            results['linreg_angle'][i] = reg_result['angle']
            
            # Add forecast if requested
            if self.forecast_periods > 0:
                slope = reg_result['slope']
                intercept = reg_result['intercept']
                if not np.isnan(slope) and not np.isnan(intercept):
                    forecast = intercept + slope * (self.window + self.forecast_periods - 1)
                    results['linreg_forecast'][i] = forecast
        
        # Convert results to pandas Series
        for key in results:
            results[key] = pd.Series(results[key], index=data.index)
        
        # Integrate regression angle as momentum measure
        results['linreg_momentum'] = results['linreg_angle'] * (100 / 45)
        results['linreg_momentum'] = results['linreg_momentum'].clip(-100, 100)
            
        self.results = results
        return results


class LinearRegressionChannelAnalyzer(AdvancedAnalysisBase):
    """
    Linear Regression Channel Analyzer
    
    Extends the linear regression line with parallel channels based on standard deviations
    from the regression line, useful for identifying potential support/resistance areas.
    """
    
    def __init__(
        self,
        name: str = "LinearRegressionChannel",
        window: int = 20,
        channel_width: Union[float, str] = "auto",
        num_std: float = 2.0,
        column: str = "close",
        **kwargs
    ):
        """Initialize the Linear Regression Channel analyzer.
        
        Args:
            name: Identifier for this analyzer
            window: Lookback period for regression calculations
            channel_width: Width of channel ('auto' for standard deviation, or fixed value)
            num_std: Number of standard deviations for channel width if 'auto'
            column: Data column to use for calculations
            **kwargs: Additional parameters
        """
        super().__init__(name=name, **kwargs)
        self.window = window
        self.channel_width = channel_width
        self.num_std = num_std
        self.column = column
        
    def _calculate_regression_channel(self, y: pd.Series) -> Dict[str, Any]:
        """Calculate linear regression channel for a given series."""
        n = len(y)
        if n <= 1:
            return {
                'middle': np.full(n, np.nan),
                'upper': np.full(n, np.nan),
                'lower': np.full(n, np.nan),
                'width': np.nan
            }
        
        x = np.arange(n)
        
        # Calculate linear regression using scipy.stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate regression line (middle of channel)
        middle = intercept + slope * x
        
        # Calculate channel width
        if self.channel_width == "auto":
            # Calculate standard deviation of residuals
            residuals = y - middle
            std_residuals = np.std(residuals)
            width = std_residuals * self.num_std
        else:
            # Use fixed width
            width = float(self.channel_width)
        
        # Calculate upper and lower channel lines
        upper = middle + width
        lower = middle - width
        
        return {
            'middle': middle,
            'upper': upper,
            'lower': lower,
            'width': width,
            'slope': slope,
            'intercept': intercept,
            'r2': r_value**2
        }
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Linear Regression Channel.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with Linear Regression Channel results
        """
        if self.column not in data.columns:
            raise ValueError(f"Data must contain '{self.column}' column")
        
        n = len(data)
        results = {
            'lrc_middle': np.full(n, np.nan),
            'lrc_upper': np.full(n, np.nan),
            'lrc_lower': np.full(n, np.nan),
            'lrc_breakout': np.zeros(n),
            'lrc_touch': np.zeros(n)
        }
        
        # Apply rolling regression channel
        for i in range(self.window - 1, n):
            window_slice = data[self.column].iloc[i - self.window + 1:i + 1].values
            reg_result = self._calculate_regression_channel(window_slice)
            
            results['lrc_middle'][i] = reg_result['middle'][-1]
            results['lrc_upper'][i] = reg_result['upper'][-1]
            results['lrc_lower'][i] = reg_result['lower'][-1]
            
        # Convert arrays to pandas Series
        for key in ['lrc_middle', 'lrc_upper', 'lrc_lower']:
            results[key] = pd.Series(results[key], index=data.index)
            
        # Add alerts for channel breakouts and touches
        for i in range(self.window, n):
            current_price = data[self.column].iloc[i]
            current_upper = results['lrc_upper'][i]
            current_lower = results['lrc_lower'][i]
            
            # Check for breakouts
            if current_price > current_upper:
                results['lrc_breakout'][i] = 1
            elif current_price < current_lower:
                results['lrc_breakout'][i] = -1
            
            # Check for channel boundary touches (within 0.1% of boundary)
            upper_touch_threshold = current_upper * 0.999
            lower_touch_threshold = current_lower * 1.001
            
            if current_price >= upper_touch_threshold and current_price <= current_upper:
                results['lrc_touch'][i] = 1
            elif current_price <= lower_touch_threshold and current_price >= current_lower:
                results['lrc_touch'][i] = -1
                
        # Convert remaining arrays to pandas Series
        for key in ['lrc_breakout', 'lrc_touch']:
            results[key] = pd.Series(results[key], index=data.index)
            
        # Add trading signal based on channel touches
        results['lrc_signal'] = pd.Series(np.zeros(n), index=data.index)
        results.loc[results['lrc_touch'] == 1, 'lrc_signal'] = -1  # Sell at upper touch
        results.loc[results['lrc_touch'] == -1, 'lrc_signal'] = 1  # Buy at lower touch
            
        self.results = results
        return results
