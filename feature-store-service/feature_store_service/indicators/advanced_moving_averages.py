"""
Advanced Moving Averages Module.

Contains implementations for:
- Triple Exponential Moving Average (TEMA)
- Double Exponential Moving Average (DEMA)
- Hull Moving Average (HullMA)
- Kaufman Adaptive Moving Average (KAMA)
- Zero-Lag Exponential Moving Average (ZLEMA)
- Arnaud Legoux Moving Average (ALMA)
- Jurik Moving Average (JMA) - Placeholder, requires specific Jurik research implementation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from feature_store_service.indicators.base_indicator import BaseIndicator

def ema(series: pd.Series, period: int) -> pd.Series:
    """Helper function for Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()

def wma(series: pd.Series, period: int) -> pd.Series:
    """Helper function for Weighted Moving Average."""
    weights = np.arange(1, period + 1)
    return series.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

class TripleExponentialMovingAverage(BaseIndicator):
    """
    Triple Exponential Moving Average (TEMA).

    Reduces lag by applying EMA calculations multiple times.
    TEMA = (3 * EMA1) - (3 * EMA2) + EMA3
    where EMA1 = EMA(data, period)
          EMA2 = EMA(EMA1, period)
          EMA3 = EMA(EMA2, period)

    Parameters:
    -----------
    period : int, optional
        The lookback period for the EMA calculations (default: 14).
    source_col : str, optional
        The data column to calculate the TEMA on (default: 'close').

    Attributes:
    -----------
    period : int
        The lookback period.
    source_col : str
        The source data column.
    """
    category = "moving_average"
    default_params = {
        "period": {"type": "int", "min": 2, "max": 200, "default": 14},
        "source_col": {"type": "str", "options": ["open", "high", "low", "close"], "default": "close"}
    }

    def __init__(self, period: int = 14, source_col: str = 'close', **kwargs):
    """
      init  .
    
    Args:
        period: Description of period
        source_col: Description of source_col
        kwargs: Description of kwargs
    
    """

        self.name = f"TEMA_{source_col}_{period}"
        self.period = period
        self.source_col = source_col
        self._validate_params()

    def _validate_params(self):
    """
     validate params.
    
    """

        if not isinstance(self.period, int) or self.period <= 1:
            raise ValueError(f"Period must be an integer greater than 1, got {self.period}")
        if not isinstance(self.source_col, str) or not self.source_col:
            raise ValueError("source_col must be a non-empty string")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate.
    
    Args:
        data: Description of data
    
    Returns:
        pd.DataFrame: Description of return value
    
    """

        self.validate_input(data, [self.source_col])
        source_data = data[self.source_col]

        ema1 = ema(source_data, self.period)
        ema2 = ema(ema1, self.period)
        ema3 = ema(ema2, self.period)

        tema = (3 * ema1) - (3 * ema2) + ema3

        output = pd.DataFrame(index=data.index)
        output[f'TEMA_{self.period}'] = tema
        return output

class DoubleExponentialMovingAverage(BaseIndicator):
    """
    Double Exponential Moving Average (DEMA).

    Similar to TEMA but uses two EMA calculations.
    DEMA = (2 * EMA1) - EMA2
    where EMA1 = EMA(data, period)
          EMA2 = EMA(EMA1, period)

    Parameters:
    -----------
    period : int, optional
        The lookback period for the EMA calculations (default: 14).
    source_col : str, optional
        The data column to calculate the DEMA on (default: 'close').

    Attributes:
    -----------
    period : int
        The lookback period.
    source_col : str
        The source data column.
    """
    category = "moving_average"
    default_params = {
        "period": {"type": "int", "min": 2, "max": 200, "default": 14},
        "source_col": {"type": "str", "options": ["open", "high", "low", "close"], "default": "close"}
    }

    def __init__(self, period: int = 14, source_col: str = 'close', **kwargs):
    """
      init  .
    
    Args:
        period: Description of period
        source_col: Description of source_col
        kwargs: Description of kwargs
    
    """

        self.name = f"DEMA_{source_col}_{period}"
        self.period = period
        self.source_col = source_col
        self._validate_params()

    def _validate_params(self):
    """
     validate params.
    
    """

        if not isinstance(self.period, int) or self.period <= 1:
            raise ValueError(f"Period must be an integer greater than 1, got {self.period}")
        if not isinstance(self.source_col, str) or not self.source_col:
            raise ValueError("source_col must be a non-empty string")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate.
    
    Args:
        data: Description of data
    
    Returns:
        pd.DataFrame: Description of return value
    
    """

        self.validate_input(data, [self.source_col])
        source_data = data[self.source_col]

        ema1 = ema(source_data, self.period)
        ema2 = ema(ema1, self.period)

        dema = (2 * ema1) - ema2

        output = pd.DataFrame(index=data.index)
        output[f'DEMA_{self.period}'] = dema
        return output

class HullMovingAverage(BaseIndicator):
    """
    Hull Moving Average (HullMA).

    A responsive and smooth moving average.
    HullMA = WMA(2 * WMA(data, period/2) - WMA(data, period), sqrt(period))

    Parameters:
    -----------
    period : int, optional
        The lookback period for the HullMA calculation (default: 9).
        Must be an integer >= 2.
    source_col : str, optional
        The data column to calculate the HullMA on (default: 'close').

    Attributes:
    -----------
    period : int
        The lookback period.
    source_col : str
        The source data column.
    """
    category = "moving_average"
    default_params = {
        "period": {"type": "int", "min": 2, "max": 200, "default": 9},
        "source_col": {"type": "str", "options": ["open", "high", "low", "close"], "default": "close"}
    }

    def __init__(self, period: int = 9, source_col: str = 'close', **kwargs):
    """
      init  .
    
    Args:
        period: Description of period
        source_col: Description of source_col
        kwargs: Description of kwargs
    
    """

        self.name = f"HullMA_{source_col}_{period}"
        self.period = period
        self.source_col = source_col
        self._validate_params()

    def _validate_params(self):
    """
     validate params.
    
    """

        if not isinstance(self.period, int) or self.period < 2:
            raise ValueError(f"Period must be an integer >= 2, got {self.period}")
        if not isinstance(self.source_col, str) or not self.source_col:
            raise ValueError("source_col must be a non-empty string")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate.
    
    Args:
        data: Description of data
    
    Returns:
        pd.DataFrame: Description of return value
    
    """

        self.validate_input(data, [self.source_col])
        source_data = data[self.source_col]

        period_half = int(self.period / 2)
        period_sqrt = int(np.sqrt(self.period))

        if period_half < 1 or period_sqrt < 1:
             raise ValueError(f"Calculated periods (half={period_half}, sqrt={period_sqrt}) must be >= 1. Increase the base period.")

        wma_half = wma(source_data, period_half)
        wma_full = wma(source_data, self.period)

        hullma_raw = (2 * wma_half) - wma_full
        hullma = wma(hullma_raw, period_sqrt)

        output = pd.DataFrame(index=data.index)
        output[f'HullMA_{self.period}'] = hullma
        return output

class KaufmanAdaptiveMovingAverage(BaseIndicator):
    """
    Kaufman Adaptive Moving Average (KAMA).

    Adjusts its smoothing based on market volatility.

    Parameters:
    -----------
    period : int, optional
        The lookback period for calculating the Efficiency Ratio (default: 10).
    fast_ema_period : int, optional
        The period for the fastest EMA used in the smoothing constant (default: 2).
    slow_ema_period : int, optional
        The period for the slowest EMA used in the smoothing constant (default: 30).
    source_col : str, optional
        The data column to calculate KAMA on (default: 'close').

    Attributes:
    -----------
    period : int
        The ER lookback period.
    fast_ema_period : int
        Fast EMA period.
    slow_ema_period : int
        Slow EMA period.
    source_col : str
        The source data column.
    """
    category = "moving_average"
    default_params = {
        "period": {"type": "int", "min": 2, "max": 200, "default": 10},
        "fast_ema_period": {"type": "int", "min": 2, "max": 50, "default": 2},
        "slow_ema_period": {"type": "int", "min": 5, "max": 200, "default": 30},
        "source_col": {"type": "str", "options": ["open", "high", "low", "close"], "default": "close"}
    }

    def __init__(self, period: int = 10, fast_ema_period: int = 2, slow_ema_period: int = 30, source_col: str = 'close', **kwargs):
    """
      init  .
    
    Args:
        period: Description of period
        fast_ema_period: Description of fast_ema_period
        slow_ema_period: Description of slow_ema_period
        source_col: Description of source_col
        kwargs: Description of kwargs
    
    """

        self.name = f"KAMA_{source_col}_{period}_{fast_ema_period}_{slow_ema_period}"
        self.period = period
        self.fast_ema_period = fast_ema_period
        self.slow_ema_period = slow_ema_period
        self.source_col = source_col
        self._validate_params()

    def _validate_params(self):
    """
     validate params.
    
    """

        if not isinstance(self.period, int) or self.period <= 1:
            raise ValueError(f"Period must be an integer greater than 1, got {self.period}")
        if not isinstance(self.fast_ema_period, int) or self.fast_ema_period <= 1:
            raise ValueError(f"Fast EMA period must be an integer greater than 1, got {self.fast_ema_period}")
        if not isinstance(self.slow_ema_period, int) or self.slow_ema_period <= self.fast_ema_period:
            raise ValueError(f"Slow EMA period must be > Fast EMA period, got slow={self.slow_ema_period}, fast={self.fast_ema_period}")
        if not isinstance(self.source_col, str) or not self.source_col:
            raise ValueError("source_col must be a non-empty string")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate.
    
    Args:
        data: Description of data
    
    Returns:
        pd.DataFrame: Description of return value
    
    """

        self.validate_input(data, [self.source_col])
        price = data[self.source_col]

        # Calculate Efficiency Ratio (ER)
        change = abs(price.diff(self.period))
        volatility = abs(price.diff(1)).rolling(window=self.period).sum()
        # Avoid division by zero
        er = (change / volatility.replace(0, np.nan)).fillna(0) # If volatility is 0, ER is 0

        # Calculate Smoothing Constant (SC)
        sc_fast = 2 / (self.fast_ema_period + 1)
        sc_slow = 2 / (self.slow_ema_period + 1)
        smoothing_constant = (er * (sc_fast - sc_slow) + sc_slow) ** 2

        # Calculate KAMA iteratively
        kama = pd.Series(index=price.index, dtype=float)
        kama.iloc[self.period - 1] = price.iloc[self.period - 1] # Start KAMA at the first available price point

        for i in range(self.period, len(price)):
            kama.iloc[i] = kama.iloc[i-1] + smoothing_constant.iloc[i] * (price.iloc[i] - kama.iloc[i-1])

        # Fill initial NaNs if necessary (though the loop starts after the first value)
        kama.fillna(method='bfill', inplace=True) # Backfill the very beginning if needed

        output = pd.DataFrame(index=data.index)
        output[f'KAMA_{self.period}_{self.fast_ema_period}_{self.slow_ema_period}'] = kama
        return output

class ZeroLagExponentialMovingAverage(BaseIndicator):
    """
    Zero-Lag Exponential Moving Average (ZLEMA).

    Aims to eliminate lag by subtracting older data influence.
    Lag = (period - 1) / 2
    ZLEMA = EMA(data + (data - data.shift(Lag)), period)

    Parameters:
    -----------
    period : int, optional
        The lookback period for the EMA calculation (default: 14).
    source_col : str, optional
        The data column to calculate the ZLEMA on (default: 'close').

    Attributes:
    -----------
    period : int
        The lookback period.
    source_col : str
        The source data column.
    """
    category = "moving_average"
    default_params = {
        "period": {"type": "int", "min": 2, "max": 200, "default": 14},
        "source_col": {"type": "str", "options": ["open", "high", "low", "close"], "default": "close"}
    }

    def __init__(self, period: int = 14, source_col: str = 'close', **kwargs):
    """
      init  .
    
    Args:
        period: Description of period
        source_col: Description of source_col
        kwargs: Description of kwargs
    
    """

        self.name = f"ZLEMA_{source_col}_{period}"
        self.period = period
        self.source_col = source_col
        self._validate_params()

    def _validate_params(self):
    """
     validate params.
    
    """

        if not isinstance(self.period, int) or self.period <= 1:
            raise ValueError(f"Period must be an integer greater than 1, got {self.period}")
        if not isinstance(self.source_col, str) or not self.source_col:
            raise ValueError("source_col must be a non-empty string")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate.
    
    Args:
        data: Description of data
    
    Returns:
        pd.DataFrame: Description of return value
    
    """

        self.validate_input(data, [self.source_col])
        source_data = data[self.source_col]

        lag = int((self.period - 1) / 2)
        if lag < 0:
             raise ValueError(f"Calculated lag ({lag}) must be >= 0. Increase the period.")

        momentum_data = source_data + (source_data.diff(lag).fillna(0))
        zlema = ema(momentum_data, self.period)

        output = pd.DataFrame(index=data.index)
        output[f'ZLEMA_{self.period}'] = zlema
        return output

class ArnaudLegouxMovingAverage(BaseIndicator):
    """
    Arnaud Legoux Moving Average (ALMA).

    Applies a Gaussian filter to the moving average, reducing lag and preserving smoothness.

    Parameters:
    -----------
    period : int, optional
        The lookback period for the ALMA (default: 9).
    sigma : float, optional
        Standard deviation for the Gaussian filter (controls smoothness, default: 6.0).
    offset : float, optional
        Offset for the Gaussian filter (controls lag, 0 to 1, default: 0.85).
    source_col : str, optional
        The data column to calculate the ALMA on (default: 'close').

    Attributes:
    -----------
    period : int
        The lookback period.
    sigma : float
        Gaussian filter standard deviation.
    offset : float
        Gaussian filter offset.
    source_col : str
        The source data column.
    """
    category = "moving_average"
    default_params = {
        "period": {"type": "int", "min": 2, "max": 200, "default": 9},
        "sigma": {"type": "float", "min": 0.1, "max": 100.0, "default": 6.0},
        "offset": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.85},
        "source_col": {"type": "str", "options": ["open", "high", "low", "close"], "default": "close"}
    }

    def __init__(self, period: int = 9, sigma: float = 6.0, offset: float = 0.85, source_col: str = 'close', **kwargs):
    """
      init  .
    
    Args:
        period: Description of period
        sigma: Description of sigma
        offset: Description of offset
        source_col: Description of source_col
        kwargs: Description of kwargs
    
    """

        self.name = f"ALMA_{source_col}_{period}_{sigma}_{offset}"
        self.period = period
        self.sigma = sigma
        self.offset = offset
        self.source_col = source_col
        self._validate_params()

    def _validate_params(self):
    """
     validate params.
    
    """

        if not isinstance(self.period, int) or self.period <= 1:
            raise ValueError(f"Period must be an integer greater than 1, got {self.period}")
        if not isinstance(self.sigma, (float, int)) or self.sigma <= 0:
            raise ValueError(f"Sigma must be positive, got {self.sigma}")
        if not isinstance(self.offset, float) or not (0.0 <= self.offset <= 1.0):
            raise ValueError(f"Offset must be a float between 0.0 and 1.0, got {self.offset}")
        if not isinstance(self.source_col, str) or not self.source_col:
            raise ValueError("source_col must be a non-empty string")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate.
    
    Args:
        data: Description of data
    
    Returns:
        pd.DataFrame: Description of return value
    
    """

        self.validate_input(data, [self.source_col])
        source_data = data[self.source_col]

        m = self.offset * (self.period - 1)
        s = self.period / self.sigma
        window = self.period

        # Calculate Gaussian weights
        weights = np.exp(-((np.arange(window) - m)**2) / (2 * s * s))
        weights_sum = weights.sum()

        # Apply weights using rolling window
        alma = source_data.rolling(window=window).apply(lambda x: np.dot(x, weights) / weights_sum, raw=True)

        output = pd.DataFrame(index=data.index)
        output[f'ALMA_{self.period}_{self.sigma}_{self.offset}'] = alma
        return output

class JurikMovingAverage(BaseIndicator):
    """
    Jurik Moving Average (JMA).

    A sophisticated adaptive moving average known for low lag and smoothness.
    The exact formula is proprietary or complex, requiring specialized implementation.
    This is a placeholder implementation.

    Parameters:
    -----------
    period : int, optional
        The primary lookback period (default: 7).
    phase : float, optional
        Phase parameter controlling lag/overshoot (-100 to 100, default: 0).
    source_col : str, optional
        The data column to calculate the JMA on (default: 'close').

    Attributes:
    -----------
    period : int
        The lookback period.
    phase : float
        Phase parameter.
    source_col : str
        The source data column.
    """
    category = "moving_average"
    default_params = {
        "period": {"type": "int", "min": 2, "max": 200, "default": 7},
        "phase": {"type": "float", "min": -100.0, "max": 100.0, "default": 0.0},
        "source_col": {"type": "str", "options": ["open", "high", "low", "close"], "default": "close"}
    }

    def __init__(self, period: int = 7, phase: float = 0.0, source_col: str = 'close', **kwargs):
    """
      init  .
    
    Args:
        period: Description of period
        phase: Description of phase
        source_col: Description of source_col
        kwargs: Description of kwargs
    
    """

        self.name = f"JMA_{source_col}_{period}_{phase}"
        self.period = period
        self.phase = phase
        self.source_col = source_col
        self._validate_params()
        import logging
        logging.warning("Jurik Moving Average (JMA) is using a placeholder (EMA) implementation due to formula complexity/proprietary nature.")

    def _validate_params(self):
    """
     validate params.
    
    """

        if not isinstance(self.period, int) or self.period <= 1:
            raise ValueError(f"Period must be an integer greater than 1, got {self.period}")
        if not isinstance(self.phase, (float, int)) or not (-100.0 <= self.phase <= 100.0):
            raise ValueError(f"Phase must be between -100.0 and 100.0, got {self.phase}")
        if not isinstance(self.source_col, str) or not self.source_col:
            raise ValueError("source_col must be a non-empty string")

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Placeholder implementation using EMA. Replace with actual JMA logic if available."""
        self.validate_input(data, [self.source_col])
        source_data = data[self.source_col]

        # --- Placeholder using EMA --- 
        # A real JMA implementation is significantly more complex involving
        # adaptive calculations based on volatility and phase.
        jma_placeholder = ema(source_data, self.period)
        # --- End Placeholder ---

        output = pd.DataFrame(index=data.index)
        output[f'JMA_{self.period}_{self.phase}'] = jma_placeholder
        return output
