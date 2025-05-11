"""
Oscillators Module.

This module provides implementations of various oscillator-type indicators.
It is designed to be used across multiple services to ensure consistent indicator implementation.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from common_lib.indicators.base_indicator import BaseIndicator


class RelativeStrengthIndex(BaseIndicator):
    """
    Relative Strength Index (RSI) indicator.

    This momentum oscillator measures the speed and change of price movements
    on a scale from 0 to 100.
    """

    category = "oscillator"
    name = "RelativeStrengthIndex"
    default_params = {"window": 14, "column": "close"}
    required_params = {"window": int, "column": str}

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Relative Strength Index indicator.

        Args:
            params: Dictionary of parameters for the indicator calculation.
                   If None, default_params will be used.
        """
        super().__init__(params)
        self.output_column = f"rsi_{self.params['window']}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with RSI values
        """
        self.validate_input(data, [self.params["column"]])

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Calculate price changes
        delta = result[self.params["column"]].diff()

        # Split gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)

        # Calculate average gains and losses
        avg_gain = gain.rolling(window=self.params["window"]).mean()
        avg_loss = loss.rolling(window=self.params["window"]).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        result[self.output_column] = 100 - (100 / (1 + rs))

        return result


class Stochastic(BaseIndicator):
    """
    Stochastic Oscillator indicator.

    This momentum indicator compares a particular closing price of a security
    to a range of its prices over a certain period of time.
    """

    category = "oscillator"
    name = "Stochastic"
    default_params = {"k_window": 14, "d_window": 3, "d_method": "sma"}
    required_params = {"k_window": int, "d_window": int, "d_method": str}

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Stochastic Oscillator indicator.

        Args:
            params: Dictionary of parameters for the indicator calculation.
                   If None, default_params will be used.
        """
        super().__init__(params)
        self.output_column_k = f"stoch_k_{self.params['k_window']}"
        self.output_column_d = f"stoch_d_{self.params['k_window']}_{self.params['d_window']}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Stochastic Oscillator values
        """
        required_cols = ['high', 'low', 'close']
        self.validate_input(data, required_cols)

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Calculate %K
        lowest_low = result['low'].rolling(window=self.params['k_window']).min()
        highest_high = result['high'].rolling(window=self.params['k_window']).max()
        result[self.output_column_k] = 100 * ((result['close'] - lowest_low) / (highest_high - lowest_low))

        # Calculate %D (signal line)
        if self.params['d_method'] == 'sma':
            result[self.output_column_d] = result[self.output_column_k].rolling(window=self.params['d_window']).mean()
        elif self.params['d_method'] == 'ema':
            result[self.output_column_d] = result[self.output_column_k].ewm(span=self.params['d_window'], adjust=False).mean()
        else:
            raise ValueError(f"Invalid d_method: {self.params['d_method']}. Expected 'sma' or 'ema'.")

        return result


class MACD(BaseIndicator):
    """
    Moving Average Convergence Divergence (MACD) indicator.

    This trend-following momentum indicator shows the relationship between
    two moving averages of a security's price.
    """

    category = "oscillator"
    name = "MACD"
    default_params = {"fast_period": 12, "slow_period": 26, "signal_period": 9, "column": "close"}
    required_params = {"fast_period": int, "slow_period": int, "signal_period": int, "column": str}

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize MACD indicator.

        Args:
            params: Dictionary of parameters for the indicator calculation.
                   If None, default_params will be used.
        """
        super().__init__(params)
        self.output_column_macd = f"macd_{self.params['fast_period']}_{self.params['slow_period']}"
        self.output_column_signal = f"macd_signal_{self.params['signal_period']}"
        self.output_column_hist = f"macd_hist_{self.params['fast_period']}_{self.params['slow_period']}_{self.params['signal_period']}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with MACD values
        """
        self.validate_input(data, [self.params["column"]])

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Calculate fast and slow EMAs
        ema_fast = result[self.params["column"]].ewm(span=self.params["fast_period"], adjust=False).mean()
        ema_slow = result[self.params["column"]].ewm(span=self.params["slow_period"], adjust=False).mean()

        # Calculate MACD line
        result[self.output_column_macd] = ema_fast - ema_slow

        # Calculate signal line
        result[self.output_column_signal] = result[self.output_column_macd].ewm(span=self.params["signal_period"], adjust=False).mean()

        # Calculate histogram
        result[self.output_column_hist] = result[self.output_column_macd] - result[self.output_column_signal]

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
    name = "CommodityChannelIndex"
    default_params = {"window": 20, "constant": 0.015}
    required_params = {"window": int, "constant": float}

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Commodity Channel Index indicator.

        Args:
            params: Dictionary of parameters for the indicator calculation.
                   If None, default_params will be used.
        """
        super().__init__(params)
        self.output_column = f"cci_{self.params['window']}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate CCI for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with CCI values
        """
        required_cols = ['high', 'low', 'close']
        self.validate_input(data, required_cols)

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Calculate Typical Price (TP)
        tp = (result['high'] + result['low'] + result['close']) / 3

        # Calculate Simple Moving Average of Typical Price
        sma_tp = tp.rolling(window=self.params['window']).mean()

        # Calculate Mean Deviation - optimized version
        # Instead of using rolling apply which can be slow, we use a vectorized approach
        # First calculate the absolute deviations
        abs_deviations = np.abs(tp - sma_tp)

        # Then calculate the mean deviation using rolling mean
        mean_dev = abs_deviations.rolling(window=self.params['window']).mean()

        # Calculate CCI
        # Avoid division by zero if mean deviation is zero
        # Use numpy where to handle division by zero more efficiently
        scaled_mean_dev = self.params['constant'] * mean_dev
        cci = np.where(
            scaled_mean_dev != 0,
            (tp - sma_tp) / scaled_mean_dev,
            0  # Default value when mean deviation is zero
        )

        # Convert back to pandas Series and handle any remaining infinities
        cci_series = pd.Series(cci, index=result.index)
        cci_series = cci_series.replace([np.inf, -np.inf], np.nan)

        result[self.output_column] = cci_series

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
    name = "WilliamsR"
    default_params = {"window": 14}
    required_params = {"window": int}

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Williams %R indicator.

        Args:
            params: Dictionary of parameters for the indicator calculation.
                   If None, default_params will be used.
        """
        super().__init__(params)
        self.output_column = f"williams_r_{self.params['window']}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Williams %R for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Williams %R values
        """
        required_cols = ['high', 'low', 'close']
        self.validate_input(data, required_cols)

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Calculate highest high and lowest low over the lookback period
        highest_high = result['high'].rolling(window=self.params['window']).max()
        lowest_low = result['low'].rolling(window=self.params['window']).min()

        # Calculate the high-low range
        hl_range = highest_high - lowest_low

        # Williams %R formula: (Highest High - Close) / (Highest High - Lowest Low) * -100
        # Use vectorized operations for better performance
        # Handle division by zero: if highest_high equals lowest_low, set to -50 (neutral)
        williams_r = np.where(
            hl_range != 0,
            ((highest_high - result['close']) / hl_range) * -100,
            -50  # Default to middle of range when there's no range
        )

        # Convert to pandas Series and handle any remaining infinities
        williams_r_series = pd.Series(williams_r, index=result.index)

        # Ensure values are within the expected range (-100 to 0)
        williams_r_series = williams_r_series.clip(lower=-100, upper=0)

        result[self.output_column] = williams_r_series

        return result


class RateOfChange(BaseIndicator):
    """
    Rate of Change (ROC) indicator.

    This momentum oscillator measures the percentage change in price over a specific
    period. It indicates the strength of a trend and can identify overbought or
    oversold conditions.
    """

    category = "oscillator"
    name = "RateOfChange"
    default_params = {"window": 10, "column": "close", "method": "percentage"}
    required_params = {"window": int, "column": str, "method": str}

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Rate of Change indicator.

        Args:
            params: Dictionary of parameters for the indicator calculation.
                   If None, default_params will be used.
        """
        super().__init__(params)
        self.output_column = f"roc_{self.params['window']}"

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ROC for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with ROC values
        """
        self.validate_input(data, [self.params["column"]])

        # Make a copy to avoid modifying the input data
        result = data.copy()

        if self.params["method"] == "percentage":
            # ROC = ((Current - Previous) / Previous) * 100
            result[self.output_column] = ((result[self.params["column"]] / result[self.params["column"]].shift(self.params["window"])) - 1) * 100
        elif self.params["method"] == "difference":
            # ROC = Current - Previous
            result[self.output_column] = result[self.params["column"]] - result[self.params["column"]].shift(self.params["window"])
        else:
            raise ValueError(f"Invalid method: {self.params['method']}. Expected 'percentage' or 'difference'.")

        return result