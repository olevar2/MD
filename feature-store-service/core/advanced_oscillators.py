"""
Advanced Oscillators Module.

Contains implementations for:
- Awesome Oscillator
- Accelerator Oscillator
- Ultimate Oscillator
- DeMarker (DeM)
- TRIX
- Know Sure Thing (KST)
- Elder Force Index (EFI)
- Relative Vigor Index (RVI)
- Fisher Transform
- Coppock Curve
- Chande Momentum Oscillator (CMO)
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from core.base_indicator import BaseIndicator

def sma(series: pd.Series, period: int) -> pd.Series:
    """Helper function for Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()

def ema(series: pd.Series, period: int) -> pd.Series:
    """Helper function for Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()

class AwesomeOscillator(BaseIndicator):
    """
    Awesome Oscillator (AO).

    Calculates the difference between a 34-period and 5-period Simple Moving Average
    of the bar's midpoints (High + Low) / 2.

    Parameters:
    -----------
    fast_period : int, optional
        The period for the fast SMA (default: 5).
    slow_period : int, optional
        The period for the slow SMA (default: 34).

    Attributes:
    -----------
    fast_period : int
        Fast SMA period.
    slow_period : int
        Slow SMA period.
    """
    category = 'oscillator'
    default_params = {'fast_period': {'type': 'int', 'min': 2, 'max': 50, 'default': 5}, 'slow_period': {'type': 'int', 'min': 10, 'max': 100, 'default': 34}}

    def __init__(self, fast_period: int=5, slow_period: int=34, **kwargs):
    """
      init  .
    
    Args:
        fast_period: Description of fast_period
        slow_period: Description of slow_period
        kwargs: Description of kwargs
    
    """

        self.name = f'AO_{fast_period}_{slow_period}'
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._validate_params()

    def _validate_params(self):
    """
     validate params.
    
    """

        if not isinstance(self.fast_period, int) or self.fast_period <= 0:
            raise ValueError(f'Fast period must be positive integer, got {self.fast_period}')
        if not isinstance(self.slow_period, int) or self.slow_period <= self.fast_period:
            raise ValueError(f'Slow period must be > Fast period, got slow={self.slow_period}, fast={self.fast_period}')

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Awesome Oscillator.

        Args:
            data (pd.DataFrame): DataFrame containing OHLCV data.
                                 Requires 'high', 'low' columns.

        Returns:
            pd.DataFrame: DataFrame with 'AO' column added.
        """
        self.validate_input(data, ['high', 'low'])
        midpoint = (data['high'] + data['low']) / 2
        fast_sma = sma(midpoint, self.fast_period)
        slow_sma = sma(midpoint, self.slow_period)
        ao = fast_sma - slow_sma
        output = pd.DataFrame(index=data.index)
        output['AO'] = ao
        return output

class AcceleratorOscillator(BaseIndicator):
    """
    Accelerator Oscillator (AC).

    Measures the acceleration of the market driving force (Awesome Oscillator).
    AC = AO - SMA(AO, 5)

    Parameters:
    -----------
    ao_fast_period : int, optional
        Fast period for the underlying Awesome Oscillator (default: 5).
    ao_slow_period : int, optional
        Slow period for the underlying Awesome Oscillator (default: 34).
    sma_period : int, optional
        Period for the SMA of the Awesome Oscillator (default: 5).

    Attributes:
    -----------
    ao_fast_period : int
        AO fast period.
    ao_slow_period : int
        AO slow period.
    sma_period : int
        SMA period for AO.
    """
    category = 'oscillator'
    default_params = {'ao_fast_period': {'type': 'int', 'min': 2, 'max': 50, 'default': 5}, 'ao_slow_period': {'type': 'int', 'min': 10, 'max': 100, 'default': 34}, 'sma_period': {'type': 'int', 'min': 2, 'max': 50, 'default': 5}}

    def __init__(self, ao_fast_period: int=5, ao_slow_period: int=34, sma_period: int=5, **kwargs):
    """
      init  .
    
    Args:
        ao_fast_period: Description of ao_fast_period
        ao_slow_period: Description of ao_slow_period
        sma_period: Description of sma_period
        kwargs: Description of kwargs
    
    """

        self.name = f'AC_{ao_fast_period}_{ao_slow_period}_{sma_period}'
        self.ao_fast_period = ao_fast_period
        self.ao_slow_period = ao_slow_period
        self.sma_period = sma_period
        self._validate_params()

    def _validate_params(self):
    """
     validate params.
    
    """

        if not isinstance(self.ao_fast_period, int) or self.ao_fast_period <= 0:
            raise ValueError(f'AO Fast period must be positive integer, got {self.ao_fast_period}')
        if not isinstance(self.ao_slow_period, int) or self.ao_slow_period <= self.ao_fast_period:
            raise ValueError(f'AO Slow period must be > AO Fast period, got slow={self.ao_slow_period}, fast={self.ao_fast_period}')
        if not isinstance(self.sma_period, int) or self.sma_period <= 0:
            raise ValueError(f'SMA period must be positive integer, got {self.sma_period}')

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Accelerator Oscillator.

        Args:
            data (pd.DataFrame): DataFrame containing OHLCV data.
                                 Requires 'high', 'low' columns.

        Returns:
            pd.DataFrame: DataFrame with 'AC' column added.
        """
        self.validate_input(data, ['high', 'low'])
        ao_indicator = AwesomeOscillator(fast_period=self.ao_fast_period, slow_period=self.ao_slow_period)
        ao_df = ao_indicator.calculate(data)
        ao = ao_df['AO']
        ao_sma = sma(ao, self.sma_period)
        ac = ao - ao_sma
        output = pd.DataFrame(index=data.index)
        output['AC'] = ac
        return output

class UltimateOscillatorIndicator(BaseIndicator):
    """
    Ultimate Oscillator (UO).

    Combines momentum across three different timeframes.

    Parameters:
    -----------
    short_period : int, optional
        Short lookback period (default: 7).
    medium_period : int, optional
        Medium lookback period (default: 14).
    long_period : int, optional
        Long lookback period (default: 28).

    Attributes:
    -----------
    short_period : int
        Short lookback period.
    medium_period : int
        Medium lookback period.
    long_period : int
        Long lookback period.
    """
    category = 'oscillator'
    default_params = {'short_period': {'type': 'int', 'min': 2, 'max': 50, 'default': 7}, 'medium_period': {'type': 'int', 'min': 5, 'max': 100, 'default': 14}, 'long_period': {'type': 'int', 'min': 10, 'max': 200, 'default': 28}}

    def __init__(self, short_period: int=7, medium_period: int=14, long_period: int=28, **kwargs):
    """
      init  .
    
    Args:
        short_period: Description of short_period
        medium_period: Description of medium_period
        long_period: Description of long_period
        kwargs: Description of kwargs
    
    """

        self.name = f'UO_{short_period}_{medium_period}_{long_period}'
        self.short_period = short_period
        self.medium_period = medium_period
        self.long_period = long_period
        self._validate_params()

    def _validate_params(self):
    """
     validate params.
    
    """

        if not isinstance(self.short_period, int) or self.short_period <= 0:
            raise ValueError(f'Short period must be positive integer, got {self.short_period}')
        if not isinstance(self.medium_period, int) or self.medium_period <= self.short_period:
            raise ValueError(f'Medium period must be > Short period, got med={self.medium_period}, short={self.short_period}')
        if not isinstance(self.long_period, int) or self.long_period <= self.medium_period:
            raise ValueError(f'Long period must be > Medium period, got long={self.long_period}, med={self.medium_period}')

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Ultimate Oscillator.

        Args:
            data (pd.DataFrame): DataFrame containing OHLCV data.
                                 Requires 'high', 'low', 'close' columns.

        Returns:
            pd.DataFrame: DataFrame with 'UO' column added.
        """
        self.validate_input(data, ['high', 'low', 'close'])
        low = data['low']
        high = data['high']
        close = data['close']
        close_prev = close.shift(1)
        bp = close - pd.concat([low, close_prev], axis=1).min(axis=1)
        tr = pd.concat([high, close_prev], axis=1).max(axis=1) - pd.concat([low, close_prev], axis=1).min(axis=1)
        bp_sum_short = bp.rolling(window=self.short_period).sum()
        tr_sum_short = tr.rolling(window=self.short_period).sum()
        bp_sum_medium = bp.rolling(window=self.medium_period).sum()
        tr_sum_medium = tr.rolling(window=self.medium_period).sum()
        bp_sum_long = bp.rolling(window=self.long_period).sum()
        tr_sum_long = tr.rolling(window=self.long_period).sum()
        avg_short = (bp_sum_short / tr_sum_short.replace(0, np.nan)).fillna(0)
        avg_medium = (bp_sum_medium / tr_sum_medium.replace(0, np.nan)).fillna(0)
        avg_long = (bp_sum_long / tr_sum_long.replace(0, np.nan)).fillna(0)
        uo = 100 * (4 * avg_short + 2 * avg_medium + avg_long) / (4 + 2 + 1)
        output = pd.DataFrame(index=data.index)
        output['UO'] = uo
        return output

class DeMarker(BaseIndicator):
    """
    DeMarker (DeM) Indicator.

    Compares the most recent high/low prices to the previous period's high/low prices
    to measure demand.

    Parameters:
    -----------
    period : int, optional
        The lookback period (default: 14).

    Attributes:
    -----------
    period : int
        The lookback period.
    """
    category = 'oscillator'
    default_params = {'period': {'type': 'int', 'min': 2, 'max': 200, 'default': 14}}

    def __init__(self, period: int=14, **kwargs):
    """
      init  .
    
    Args:
        period: Description of period
        kwargs: Description of kwargs
    
    """

        self.name = f'DeM_{period}'
        self.period = period
        self._validate_params()

    def _validate_params(self):
    """
     validate params.
    
    """

        if not isinstance(self.period, int) or self.period <= 1:
            raise ValueError(f'Period must be an integer greater than 1, got {self.period}')

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the DeMarker Indicator.

        Args:
            data (pd.DataFrame): DataFrame containing OHLCV data.
                                 Requires 'high', 'low' columns.

        Returns:
            pd.DataFrame: DataFrame with 'DeM' column added.
        """
        self.validate_input(data, ['high', 'low'])
        high = data['high']
        low = data['low']
        high_prev = high.shift(1)
        low_prev = low.shift(1)
        demax = pd.Series(np.where(high > high_prev, high - high_prev, 0), index=data.index)
        demin = pd.Series(np.where(low < low_prev, low_prev - low, 0), index=data.index)
        sma_demax = sma(demax, self.period)
        sma_demin = sma(demin, self.period)
        demarker_denominator = (sma_demax + sma_demin).replace(0, np.nan)
        demarker = (sma_demax / demarker_denominator).fillna(0.5)
        output = pd.DataFrame(index=data.index)
        output['DeM'] = demarker
        return output

class TRIXIndicatorImpl(BaseIndicator):
    """
    TRIX Indicator.

    Measures the percentage rate of change of a triple exponentially smoothed moving average.

    Parameters:
    -----------
    period : int, optional
        The lookback period for the EMA calculations (default: 15).
    signal_period : int, optional
        The lookback period for the signal line (EMA of TRIX) (default: 9).
    source_col : str, optional
        The data column to calculate TRIX on (default: 'close').

    Attributes:
    -----------
    period : int
        The EMA lookback period.
    signal_period : int
        The signal line EMA period.
    source_col : str
        The source data column.
    """
    category = 'oscillator'
    default_params = {'period': {'type': 'int', 'min': 2, 'max': 200, 'default': 15}, 'signal_period': {'type': 'int', 'min': 2, 'max': 100, 'default': 9}, 'source_col': {'type': 'str', 'options': ['open', 'high', 'low', 'close'], 'default': 'close'}}

    def __init__(self, period: int=15, signal_period: int=9, source_col: str='close', **kwargs):
    """
      init  .
    
    Args:
        period: Description of period
        signal_period: Description of signal_period
        source_col: Description of source_col
        kwargs: Description of kwargs
    
    """

        self.name = f'TRIX_{source_col}_{period}_{signal_period}'
        self.period = period
        self.signal_period = signal_period
        self.source_col = source_col
        self._validate_params()

    def _validate_params(self):
    """
     validate params.
    
    """

        if not isinstance(self.period, int) or self.period <= 1:
            raise ValueError(f'Period must be an integer greater than 1, got {self.period}')
        if not isinstance(self.signal_period, int) or self.signal_period <= 1:
            raise ValueError(f'Signal period must be an integer greater than 1, got {self.signal_period}')
        if not isinstance(self.source_col, str) or not self.source_col:
            raise ValueError('source_col must be a non-empty string')

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the TRIX Indicator and its signal line.

        Args:
            data (pd.DataFrame): DataFrame containing the source column data.
                                 Requires column specified in `source_col`.

        Returns:
            pd.DataFrame: DataFrame with 'TRIX' and 'TRIX_Signal' columns added.
        """
        self.validate_input(data, [self.source_col])
        source_data = data[self.source_col]
        ema1 = ema(source_data, self.period)
        ema2 = ema(ema1, self.period)
        ema3 = ema(ema2, self.period)
        trix = ema3.pct_change(1) * 100
        trix_signal = ema(trix, self.signal_period)
        output = pd.DataFrame(index=data.index)
        output['TRIX'] = trix
        output[f'TRIX_Signal_{self.signal_period}'] = trix_signal
        return output

class KSTIndicatorImpl(BaseIndicator):
    """
    Know Sure Thing (KST) Indicator.

    A momentum oscillator based on the smoothed rate-of-change over four different time periods.

    Parameters:
    -----------
    roc_period1, roc_period2, roc_period3, roc_period4 : int, optional
        Periods for the four Rate of Change calculations (defaults: 10, 15, 20, 30).
    sma_period1, sma_period2, sma_period3, sma_period4 : int, optional
        Periods for the SMA smoothing of each ROC (defaults: 10, 10, 10, 15).
    signal_period : int, optional
        Period for the signal line (SMA of KST) (default: 9).
    source_col : str, optional
        The data column to calculate KST on (default: 'close').

    Attributes:
    -----------
    (All parameters)
    """
    category = 'oscillator'
    default_params = {'roc_period1': {'type': 'int', 'min': 1, 'max': 50, 'default': 10}, 'roc_period2': {'type': 'int', 'min': 1, 'max': 50, 'default': 15}, 'roc_period3': {'type': 'int', 'min': 1, 'max': 50, 'default': 20}, 'roc_period4': {'type': 'int', 'min': 1, 'max': 50, 'default': 30}, 'sma_period1': {'type': 'int', 'min': 1, 'max': 50, 'default': 10}, 'sma_period2': {'type': 'int', 'min': 1, 'max': 50, 'default': 10}, 'sma_period3': {'type': 'int', 'min': 1, 'max': 50, 'default': 10}, 'sma_period4': {'type': 'int', 'min': 1, 'max': 50, 'default': 15}, 'signal_period': {'type': 'int', 'min': 1, 'max': 50, 'default': 9}, 'source_col': {'type': 'str', 'options': ['open', 'high', 'low', 'close'], 'default': 'close'}}

    def __init__(self, roc_period1: int=10, roc_period2: int=15, roc_period3: int=20, roc_period4: int=30, sma_period1: int=10, sma_period2: int=10, sma_period3: int=10, sma_period4: int=15, signal_period: int=9, source_col: str='close', **kwargs):
    """
      init  .
    
    Args:
        roc_period1: Description of roc_period1
        roc_period2: Description of roc_period2
        roc_period3: Description of roc_period3
        roc_period4: Description of roc_period4
        sma_period1: Description of sma_period1
        sma_period2: Description of sma_period2
        sma_period3: Description of sma_period3
        sma_period4: Description of sma_period4
        signal_period: Description of signal_period
        source_col: Description of source_col
        kwargs: Description of kwargs
    
    """

        self.name = f'KST_{source_col}_{roc_period1}_{roc_period2}_{roc_period3}_{roc_period4}_{sma_period1}_{sma_period2}_{sma_period3}_{sma_period4}_{signal_period}'
        self.roc_period1 = roc_period1
        self.roc_period2 = roc_period2
        self.roc_period3 = roc_period3
        self.roc_period4 = roc_period4
        self.sma_period1 = sma_period1
        self.sma_period2 = sma_period2
        self.sma_period3 = sma_period3
        self.sma_period4 = sma_period4
        self.signal_period = signal_period
        self.source_col = source_col
        self._validate_params()

    def _validate_params(self):
    """
     validate params.
    
    """

        periods = [self.roc_period1, self.roc_period2, self.roc_period3, self.roc_period4, self.sma_period1, self.sma_period2, self.sma_period3, self.sma_period4, self.signal_period]
        for p in periods:
            if not isinstance(p, int) or p <= 0:
                raise ValueError(f'All KST periods must be positive integers, got {p}')
        if not isinstance(self.source_col, str) or not self.source_col:
            raise ValueError('source_col must be a non-empty string')

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Know Sure Thing (KST) Indicator and its signal line.

        Args:
            data (pd.DataFrame): DataFrame containing the source column data.
                                 Requires column specified in `source_col`.

        Returns:
            pd.DataFrame: DataFrame with 'KST' and 'KST_Signal' columns added.
        """
        self.validate_input(data, [self.source_col])
        price = data[self.source_col]
        roc1 = price.pct_change(self.roc_period1)
        roc2 = price.pct_change(self.roc_period2)
        roc3 = price.pct_change(self.roc_period3)
        roc4 = price.pct_change(self.roc_period4)
        sma_roc1 = sma(roc1, self.sma_period1)
        sma_roc2 = sma(roc2, self.sma_period2)
        sma_roc3 = sma(roc3, self.sma_period3)
        sma_roc4 = sma(roc4, self.sma_period4)
        kst = sma_roc1 * 1 + sma_roc2 * 2 + sma_roc3 * 3 + sma_roc4 * 4
        kst_signal = sma(kst, self.signal_period)
        output = pd.DataFrame(index=data.index)
        output['KST'] = kst
        output[f'KST_Signal_{self.signal_period}'] = kst_signal
        return output

class ElderForceIndex(BaseIndicator):
    """
    Elder Force Index (EFI).

    Measures the power behind a price movement using price change and volume.
    EFI = (Current Close - Previous Close) * Current Volume
    Smoothed EFI = EMA(EFI, period)

    Parameters:
    -----------
    period : int, optional
        The lookback period for the EMA smoothing (default: 13).

    Attributes:
    -----------
    period : int
        The EMA lookback period.
    """
    category = 'oscillator'
    default_params = {'period': {'type': 'int', 'min': 2, 'max': 200, 'default': 13}}

    def __init__(self, period: int=13, **kwargs):
    """
      init  .
    
    Args:
        period: Description of period
        kwargs: Description of kwargs
    
    """

        self.name = f'EFI_{period}'
        self.period = period
        self._validate_params()

    def _validate_params(self):
    """
     validate params.
    
    """

        if not isinstance(self.period, int) or self.period <= 1:
            raise ValueError(f'Period must be an integer greater than 1, got {self.period}')

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Elder Force Index.

        Args:
            data (pd.DataFrame): DataFrame containing OHLCV data.
                                 Requires 'close', 'volume' columns.

        Returns:
            pd.DataFrame: DataFrame with 'EFI' column added.
        """
        self.validate_input(data, ['close', 'volume'])
        close = data['close']
        volume = data['volume']
        close_diff = close.diff(1)
        force_index_1 = close_diff * volume
        efi = ema(force_index_1, self.period)
        output = pd.DataFrame(index=data.index)
        output['EFI'] = efi
        return output

class RelativeVigorIndex(BaseIndicator):
    """
    Relative Vigor Index (RVI).

    Compares the closing price to the trading range, smoothed with moving averages.

    Parameters:
    -----------
    period : int, optional
        The lookback period for smoothing (default: 10).
    signal_period : int, optional
        The lookback period for the signal line (default: 4, uses symmetrical weighted MA).

    Attributes:
    -----------
    period : int
        The smoothing lookback period.
    signal_period : int
        The signal line lookback period.
    """
    category = 'oscillator'
    default_params = {'period': {'type': 'int', 'min': 2, 'max': 200, 'default': 10}, 'signal_period': {'type': 'int', 'min': 2, 'max': 50, 'default': 4}}

    def __init__(self, period: int=10, signal_period: int=4, **kwargs):
    """
      init  .
    
    Args:
        period: Description of period
        signal_period: Description of signal_period
        kwargs: Description of kwargs
    
    """

        self.name = f'RVI_{period}_{signal_period}'
        self.period = period
        self.signal_period = signal_period
        self._validate_params()

    def _validate_params(self):
    """
     validate params.
    
    """

        if not isinstance(self.period, int) or self.period <= 1:
            raise ValueError(f'Period must be an integer greater than 1, got {self.period}')
        if not isinstance(self.signal_period, int) or self.signal_period <= 1:
            raise ValueError(f'Signal period must be an integer greater than 1, got {self.signal_period}')

    def _sym_weighted_ma(self, series: pd.Series, period: int) -> pd.Series:
        """Symmetrically Weighted Moving Average (1-2-2-1 for period 4)."""
        if period != 4:
            import logging
            logging.warning(f'RVI Signal Line using SMA for period {period}, symmetrical weights only defined for period 4.')
            return sma(series, period)
        weights = np.array([1, 2, 2, 1])
        return series.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Relative Vigor Index (RVI) and its signal line.

        Args:
            data (pd.DataFrame): DataFrame containing OHLCV data.
                                 Requires 'open', 'high', 'low', 'close' columns.

        Returns:
            pd.DataFrame: DataFrame with 'RVI' and 'RVI_Signal' columns added.
        """
        self.validate_input(data, ['open', 'high', 'low', 'close'])
        close = data['close']
        open = data['open']
        high = data['high']
        low = data['low']
        numerator = close - open + 2 * (close.shift(1) - open.shift(1)) + 2 * (close.shift(2) - open.shift(2)) + (close.shift(3) - open.shift(3))
        denominator = high - low + 2 * (high.shift(1) - low.shift(1)) + 2 * (high.shift(2) - low.shift(2)) + (high.shift(3) - low.shift(3))
        sum_numerator = numerator.rolling(window=self.period).sum()
        sum_denominator = denominator.rolling(window=self.period).sum()
        rvi = (sum_numerator / sum_denominator.replace(0, np.nan)).fillna(method='ffill').fillna(0)
        rvi_signal = self._sym_weighted_ma(rvi, self.signal_period)
        output = pd.DataFrame(index=data.index)
        output['RVI'] = rvi
        output[f'RVI_Signal_{self.signal_period}'] = rvi_signal
        return output

class FisherTransform(BaseIndicator):
    """
    Fisher Transform Indicator.

    Transforms prices into a Gaussian normal distribution to identify extremes.

    Parameters:
    -----------
    period : int, optional
        The lookback period for finding min/max of the price transformation (default: 9).
    source_col : str, optional
        The data column to transform (typically midpoint 'hl2', default: 'close').
        If 'hl2', requires 'high' and 'low'.

    Attributes:
    -----------
    period : int
        The lookback period.
    source_col : str
        The source data column ('close', 'hl2', etc.).
    """
    category = 'oscillator'
    default_params = {'period': {'type': 'int', 'min': 2, 'max': 200, 'default': 9}, 'source_col': {'type': 'str', 'options': ['open', 'high', 'low', 'close', 'hl2'], 'default': 'hl2'}}

    def __init__(self, period: int=9, source_col: str='hl2', **kwargs):
    """
      init  .
    
    Args:
        period: Description of period
        source_col: Description of source_col
        kwargs: Description of kwargs
    
    """

        self.name = f'Fisher_{source_col}_{period}'
        self.period = period
        self.source_col = source_col
        self._validate_params()

    def _validate_params(self):
    """
     validate params.
    
    """

        if not isinstance(self.period, int) or self.period <= 1:
            raise ValueError(f'Period must be an integer greater than 1, got {self.period}')
        if self.source_col not in ['open', 'high', 'low', 'close', 'hl2']:
            raise ValueError(f'Invalid source_col: {self.source_col}')

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Fisher Transform.

        Args:
            data (pd.DataFrame): DataFrame containing OHLCV data.
                                 Requires columns based on `source_col`.

        Returns:
            pd.DataFrame: DataFrame with 'Fisher' and 'Fisher_Signal' columns added.
        """
        required_cols = []
        if self.source_col == 'hl2':
            required_cols = ['high', 'low']
            price = (data['high'] + data['low']) / 2
        else:
            required_cols = [self.source_col]
            price = data[self.source_col]
        self.validate_input(data, required_cols)
        min_price = price.rolling(window=self.period).min()
        max_price = price.rolling(window=self.period).max()
        price_range = (max_price - min_price).replace(0, np.nan)
        value1_raw = ((price - min_price) / price_range).fillna(0.5)
        value1 = 0.99 * (2 * value1_raw - 1)
        fisher = 0.5 * np.log((1 + value1) / (1 - value1).replace(0, 1e-10))
        fisher.replace([np.inf, -np.inf], np.nan, inplace=True)
        fisher.fillna(method='ffill', inplace=True)
        fisher.fillna(0, inplace=True)
        fisher_signal = fisher.shift(1)
        output = pd.DataFrame(index=data.index)
        output[f'Fisher_{self.period}'] = fisher
        output[f'Fisher_Signal_{self.period}'] = fisher_signal
        return output

class CoppockCurveIndicatorImpl(BaseIndicator):
    """
    Coppock Curve Indicator.

    A momentum indicator based on the sum of two weighted rates-of-change.

    Parameters:
    -----------
    long_roc_period : int, optional
        Period for the longer Rate of Change (default: 14).
    short_roc_period : int, optional
        Period for the shorter Rate of Change (default: 11).
    wma_period : int, optional
        Period for the Weighted Moving Average smoothing (default: 10).
    source_col : str, optional
        The data column to calculate Coppock Curve on (default: 'close').

    Attributes:
    -----------
    (All parameters)
    """
    category = 'oscillator'
    default_params = {'long_roc_period': {'type': 'int', 'min': 2, 'max': 50, 'default': 14}, 'short_roc_period': {'type': 'int', 'min': 2, 'max': 50, 'default': 11}, 'wma_period': {'type': 'int', 'min': 2, 'max': 50, 'default': 10}, 'source_col': {'type': 'str', 'options': ['open', 'high', 'low', 'close'], 'default': 'close'}}

    def __init__(self, long_roc_period: int=14, short_roc_period: int=11, wma_period: int=10, source_col: str='close', **kwargs):
    """
      init  .
    
    Args:
        long_roc_period: Description of long_roc_period
        short_roc_period: Description of short_roc_period
        wma_period: Description of wma_period
        source_col: Description of source_col
        kwargs: Description of kwargs
    
    """

        self.name = f'Coppock_{source_col}_{long_roc_period}_{short_roc_period}_{wma_period}'
        self.long_roc_period = long_roc_period
        self.short_roc_period = short_roc_period
        self.wma_period = wma_period
        self.source_col = source_col
        self._validate_params()

    def _validate_params(self):
    """
     validate params.
    
    """

        if not isinstance(self.long_roc_period, int) or self.long_roc_period <= 1:
            raise ValueError(f'Long ROC period must be > 1, got {self.long_roc_period}')
        if not isinstance(self.short_roc_period, int) or self.short_roc_period <= 1:
            raise ValueError(f'Short ROC period must be > 1, got {self.short_roc_period}')
        if self.short_roc_period >= self.long_roc_period:
            raise ValueError(f'Short ROC period must be < Long ROC period, got short={self.short_roc_period}, long={self.long_roc_period}')
        if not isinstance(self.wma_period, int) or self.wma_period <= 1:
            raise ValueError(f'WMA period must be > 1, got {self.wma_period}')
        if not isinstance(self.source_col, str) or not self.source_col:
            raise ValueError('source_col must be a non-empty string')

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Coppock Curve.

        Args:
            data (pd.DataFrame): DataFrame containing the source column data.
                                 Requires column specified in `source_col`.

        Returns:
            pd.DataFrame: DataFrame with 'Coppock' column added.
        """
        self.validate_input(data, [self.source_col])
        price = data[self.source_col]
        roc_long = price.pct_change(self.long_roc_period) * 100
        roc_short = price.pct_change(self.short_roc_period) * 100
        roc_sum = roc_long + roc_short
        coppock = wma(roc_sum, self.wma_period)
        output = pd.DataFrame(index=data.index)
        output['Coppock'] = coppock
        return output

class ChandeMomentumOscillator(BaseIndicator):
    """
    Chande Momentum Oscillator (CMO).

    Measures momentum by relating the sum of recent gains to the sum of recent losses.

    Parameters:
    -----------
    period : int, optional
        The lookback period (default: 9).
    source_col : str, optional
        The data column to calculate CMO on (default: 'close').

    Attributes:
    -----------
    period : int
        The lookback period.
    source_col : str
        The source data column.
    """
    category = 'oscillator'
    default_params = {'period': {'type': 'int', 'min': 2, 'max': 200, 'default': 9}, 'source_col': {'type': 'str', 'options': ['open', 'high', 'low', 'close'], 'default': 'close'}}

    def __init__(self, period: int=9, source_col: str='close', **kwargs):
    """
      init  .
    
    Args:
        period: Description of period
        source_col: Description of source_col
        kwargs: Description of kwargs
    
    """

        self.name = f'CMO_{source_col}_{period}'
        self.period = period
        self.source_col = source_col
        self._validate_params()

    def _validate_params(self):
    """
     validate params.
    
    """

        if not isinstance(self.period, int) or self.period <= 1:
            raise ValueError(f'Period must be an integer greater than 1, got {self.period}')
        if not isinstance(self.source_col, str) or not self.source_col:
            raise ValueError('source_col must be a non-empty string')

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Chande Momentum Oscillator.

        Args:
            data (pd.DataFrame): DataFrame containing the source column data.
                                 Requires column specified in `source_col`.

        Returns:
            pd.DataFrame: DataFrame with 'CMO' column added.
        """
        self.validate_input(data, [self.source_col])
        price = data[self.source_col]
        delta = price.diff(1)
        gain = delta.where(delta > 0, 0).rolling(window=self.period).sum()
        loss = abs(delta.where(delta < 0, 0)).rolling(window=self.period).sum()
        cmo_denominator = (gain + loss).replace(0, np.nan)
        cmo = 100 * (gain - loss) / cmo_denominator
        cmo.fillna(0, inplace=True)
        output = pd.DataFrame(index=data.index)
        output['CMO'] = cmo
        return output