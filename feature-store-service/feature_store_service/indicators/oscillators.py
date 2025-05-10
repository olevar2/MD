"""
Oscillators Module.

This module provides implementations of various oscillator-type indicators.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from feature_store_service.indicators.base_indicator import BaseIndicator
from feature_store_service.utils.profiling import log_and_time
from feature_store_service.caching.indicator_cache import cache_indicator


class RelativeStrengthIndex(BaseIndicator):
    """
    Relative Strength Index (RSI) indicator.

    This momentum oscillator measures the speed and change of price movements
    on a scale from 0 to 100.
    """

    category = "oscillator"

    def __init__(self, window: int = 14, column: str = "close", **kwargs):
        """
        Initialize Relative Strength Index indicator.

        Args:
            window: Lookback period for the RSI calculation
            column: Data column to use for calculations (default: 'close')
            **kwargs: Additional parameters
        """
        self.window = window
        self.column = column
        self.name = f"rsi_{window}"

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, symbol: str, timeframe: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI for the given data.

        Args:
            symbol: Symbol for the data
            timeframe: Timeframe for the data
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with RSI values
        """
        # TODO: Add unit tests comparing results with a known library (e.g., TA-Lib) for validation.
        # TODO: Optimize rolling calculations if performance issues arise on large datasets.
        if self.column not in data.columns:
            raise ValueError(f"Data must contain '{self.column}' column")

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Calculate price changes
        delta = result[self.column].diff()

        # Split gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)

        # Calculate average gains and losses
        avg_gain = gain.rolling(window=self.window).mean()
        avg_loss = loss.rolling(window=self.window).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        result[self.name] = 100 - (100 / (1 + rs))

        return result


class Stochastic(BaseIndicator):
    """
    Stochastic Oscillator indicator.

    This momentum indicator compares a particular closing price of a security
    to a range of its prices over a certain period of time.
    """

    category = "oscillator"

    def __init__(
        self,
        k_window: int = 14,
        d_window: int = 3,
        d_method: str = "sma",
        **kwargs
    ):
        """
        Initialize Stochastic Oscillator indicator.

        Args:
            k_window: Lookback period for the %K line
            d_window: Smoothing period for the %D line
            d_method: Method for calculating %D ('sma' or 'ema')
            **kwargs: Additional parameters
        """
        self.k_window = k_window
        self.d_window = d_window
        self.d_method = d_method
        self.name_k = f"stoch_k_{k_window}"
        self.name_d = f"stoch_d_{k_window}_{d_window}"

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, symbol: str, timeframe: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator for the given data.

        Args:
            symbol: Symbol for the data
            timeframe: Timeframe for the data
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Stochastic Oscillator values
        """
        # TODO: Add unit tests for both 'sma' and 'ema' d_method options.
        # TODO: Check for potential optimizations in rolling min/max calculations.
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Calculate %K
        lowest_low = result['low'].rolling(window=self.k_window).min()
        highest_high = result['high'].rolling(window=self.k_window).max()
        result[self.name_k] = 100 * ((result['close'] - lowest_low) / (highest_high - lowest_low))

        # Calculate %D (signal line)
        if self.d_method == 'sma':
            result[self.name_d] = result[self.name_k].rolling(window=self.d_window).mean()
        elif self.d_method == 'ema':
            result[self.name_d] = result[self.name_k].ewm(span=self.d_window, adjust=False).mean()
        else:
            raise ValueError(f"Invalid d_method: {self.d_method}. Expected 'sma' or 'ema'.")

        return result


class MACD(BaseIndicator):
    """
    Moving Average Convergence Divergence (MACD) indicator.

    This trend-following momentum indicator shows the relationship between
    two moving averages of a security's price.
    """

    category = "oscillator"

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = "close",
        **kwargs
    ):
        """
        Initialize MACD indicator.

        Args:
            fast_period: Period for the fast EMA
            slow_period: Period for the slow EMA
            signal_period: Period for the signal line EMA
            column: Data column to use for calculations (default: 'close')
            **kwargs: Additional parameters
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.column = column
        self.name_macd = f"macd_{fast_period}_{slow_period}"
        self.name_signal = f"macd_signal_{signal_period}"
        self.name_hist = f"macd_hist_{fast_period}_{slow_period}_{signal_period}"

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, symbol: str, timeframe: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD for the given data.

        Args:
            symbol: Symbol for the data
            timeframe: Timeframe for the data
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with MACD values
        """
        # TODO: Add unit tests comparing results against known values or another library.
        # TODO: Profile the multiple EWM calculations.
        if self.column not in data.columns:
            raise ValueError(f"Data must contain '{self.column}' column")

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Calculate fast and slow EMAs
        ema_fast = result[self.column].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = result[self.column].ewm(span=self.slow_period, adjust=False).mean()

        # Calculate MACD line
        result[self.name_macd] = ema_fast - ema_slow

        # Calculate signal line
        result[self.name_signal] = result[self.name_macd].ewm(span=self.signal_period, adjust=False).mean()

        # Calculate histogram
        result[self.name_hist] = result[self.name_macd] - result[self.name_signal]

        return result

class CommodityChannelIndex(BaseIndicator):
    """
    Commodity Channel Index (CCI) indicator.

    This momentum-based oscillator measures the current price level relative
    to an average price level over a given period.

    The CCI is calculated as:
    CCI = (Typical Price - SMA of Typical Price) / (Constant * Mean Deviation)

    Where:
    - Typical Price = (High + Low + Close) / 3
    - SMA = Simple Moving Average
    - Constant = Typically 0.015
    - Mean Deviation = Average of absolute deviations from the mean
    """

    category = "oscillator"

    def __init__(self, window: int = 20, constant: float = 0.015, **kwargs):
        """
        Initialize Commodity Channel Index indicator.

        Args:
            window: Lookback period for the CCI calculation
            constant: Scaling constant (typically 0.015)
            **kwargs: Additional parameters
        """
        self.window = window
        self.constant = constant
        self.name = f"cci_{window}"

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, symbol: str, timeframe: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate CCI for the given data.

        Args:
            symbol: Symbol for the data
            timeframe: Timeframe for the data
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with CCI values
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Calculate Typical Price (TP)
        tp = (result['high'] + result['low'] + result['close']) / 3

        # Calculate Simple Moving Average of Typical Price
        sma_tp = tp.rolling(window=self.window).mean()

        # Calculate Mean Deviation - optimized version
        # Instead of using rolling apply which can be slow, we use a vectorized approach
        # First calculate the absolute deviations
        abs_deviations = np.abs(tp - sma_tp)

        # Then calculate the mean deviation using rolling mean
        mean_dev = abs_deviations.rolling(window=self.window).mean()

        # Calculate CCI
        # Avoid division by zero if mean deviation is zero
        # Use numpy where to handle division by zero more efficiently
        scaled_mean_dev = self.constant * mean_dev
        cci = np.where(
            scaled_mean_dev != 0,
            (tp - sma_tp) / scaled_mean_dev,
            0  # Default value when mean deviation is zero
        )

        # Convert back to pandas Series and handle any remaining infinities
        cci_series = pd.Series(cci, index=result.index)
        cci_series = cci_series.replace([np.inf, -np.inf], np.nan)

        result[self.name] = cci_series

        return result

class WilliamsR(BaseIndicator):
    """
    Williams %R indicator.

    This momentum oscillator measures overbought and oversold levels,
    similar to Stochastic but without the smoothing. It shows the relationship
    of the close relative to the high-low range over a set period.

    The Williams %R is calculated as:
    %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

    Where:
    - Highest High = Maximum high price over the lookback period
    - Lowest Low = Minimum low price over the lookback period

    The indicator ranges from 0 to -100:
    - Values between -80 and -100 are considered oversold
    - Values between 0 and -20 are considered overbought
    """

    category = "oscillator"

    def __init__(self, window: int = 14, **kwargs):
        """
        Initialize Williams %R indicator.

        Args:
            window: Lookback period for calculation
            **kwargs: Additional parameters
        """
        self.window = window
        self.name = f"williams_r_{window}"

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, symbol: str, timeframe: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Williams %R for the given data.

        Args:
            symbol: Symbol for the data
            timeframe: Timeframe for the data
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Williams %R values
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Calculate highest high and lowest low over the lookback period
        # Use numba or vectorized operations for better performance on large datasets
        highest_high = result['high'].rolling(window=self.window).max()
        lowest_low = result['low'].rolling(window=self.window).min()

        # Calculate the high-low range
        hl_range = highest_high - lowest_low

        # Williams %R formula: (Highest High - Close) / (Highest High - Lowest Low) * -100
        # Use vectorized operations for better performance
        # Handle division by zero: if highest_high equals lowest_low, set to 0 (neutral)
        williams_r = np.where(
            hl_range != 0,
            ((highest_high - result['close']) / hl_range) * -100,
            -50  # Default to middle of range when there's no range
        )

        # Convert to pandas Series and handle any remaining infinities
        williams_r_series = pd.Series(williams_r, index=result.index)

        # Ensure values are within the expected range (-100 to 0)
        williams_r_series = williams_r_series.clip(lower=-100, upper=0)

        result[self.name] = williams_r_series

        return result

class RateOfChange(BaseIndicator):
    """
    Rate of Change (ROC) indicator.

    This momentum oscillator measures the percentage change in price over a specific
    period. It indicates the strength of a trend and can identify overbought or
    oversold conditions.
    """

    category = "oscillator"

    def __init__(self, window: int = 10, column: str = "close", method: str = "percentage", **kwargs):
        """
        Initialize Rate of Change indicator.

        Args:
            window: Lookback period for calculation (e.g., 10 days)
            column: Data column to use for calculations (default: 'close')
            method: Calculation method ('percentage' or 'difference')
            **kwargs: Additional parameters
        """
        self.window = window
        self.column = column
        self.method = method
        self.name = f"roc_{window}"

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, symbol: str, timeframe: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ROC for the given data.

        Args:
            symbol: Symbol for the data
            timeframe: Timeframe for the data
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with ROC values
        """
        # TODO: Add unit tests for both 'percentage' and 'difference' methods.
        # TODO: Ensure sufficient data exists for the shift operation.
        if self.column not in data.columns:
            raise ValueError(f"Data must contain '{self.column}' column")

        # Make a copy to avoid modifying the input data
        result = data.copy()

        if self.method == "percentage":
            # ROC = ((Current - Previous) / Previous) * 100
            result[self.name] = ((result[self.column] / result[self.column].shift(self.window)) - 1) * 100
        else:  # difference method
            # ROC = Current - Previous
            result[self.name] = result[self.column] - result[self.column].shift(self.window)

        return result
