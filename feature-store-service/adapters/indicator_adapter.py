"""
Indicator Adapter Module.

This module provides adapters for using common-lib indicators in the feature store service.
It ensures backward compatibility with existing code while leveraging the standardized
indicator implementations from common-lib.
"""

from typing import Dict, Any, List, Optional, Type
import pandas as pd
from common_lib.indicators.base_indicator import BaseIndicator
from common_lib.indicators.moving_averages import (
    SimpleMovingAverage as CommonSimpleMovingAverage,
    ExponentialMovingAverage as CommonExponentialMovingAverage,
    WeightedMovingAverage as CommonWeightedMovingAverage
)
from common_lib.indicators.oscillators import (
    RelativeStrengthIndex as CommonRelativeStrengthIndex,
    Stochastic as CommonStochastic,
    MACD as CommonMACD,
    CommodityChannelIndex as CommonCommodityChannelIndex,
    WilliamsR as CommonWilliamsR,
    RateOfChange as CommonRateOfChange
)
from common_lib.indicators.volatility import (
    BollingerBands as CommonBollingerBands,
    KeltnerChannels as CommonKeltnerChannels,
    DonchianChannels as CommonDonchianChannels,
    AverageTrueRange as CommonAverageTrueRange,
    PriceEnvelopes as CommonPriceEnvelopes,
    HistoricalVolatility as CommonHistoricalVolatility
)
from core.profiling import log_and_time
from core.indicator_cache import cache_indicator


class IndicatorAdapter:
    """
    Adapter for common-lib indicators.

    This class adapts common-lib indicators to the feature store service interface,
    ensuring backward compatibility with existing code.
    """

    def __init__(self, indicator: BaseIndicator):
        """
        Initialize the adapter with a common-lib indicator.

        Args:
            indicator: Common-lib indicator instance
        """
        self.indicator = indicator
        self.name = indicator.params.get("output_column", indicator.name)

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicator values using the common-lib indicator.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with indicator values
        """
        return self.indicator.calculate(data)


class SimpleMovingAverage:
    """
    Simple Moving Average (SMA) indicator adapter.

    This class provides backward compatibility with the feature store service
    while using the common-lib implementation.
    """

    category = "moving_average"

    def __init__(self, window: int = 14, column: str = "close", **kwargs):
        """
        Initialize Simple Moving Average indicator.

        Args:
            window: Lookback period for the moving average
            column: Data column to use for calculations (default: 'close')
            **kwargs: Additional parameters
        """
        self.window = window
        self.column = column
        self.name = f"sma_{window}"

        # Create common-lib indicator
        self.indicator = CommonSimpleMovingAverage({
            "window": window,
            "column": column,
            "output_column": self.name
        })

        # Create adapter
        self.adapter = IndicatorAdapter(self.indicator)

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate SMA for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with SMA values
        """
        return self.adapter.calculate(data)


class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) indicator adapter.

    This class provides backward compatibility with the feature store service
    while using the common-lib implementation.
    """

    category = "moving_average"

    def __init__(self, window: int = 14, column: str = "close", **kwargs):
        """
        Initialize Exponential Moving Average indicator.

        Args:
            window: Lookback period for the moving average
            column: Data column to use for calculations (default: 'close')
            **kwargs: Additional parameters
        """
        self.window = window
        self.column = column
        self.name = f"ema_{window}"

        # Create common-lib indicator
        self.indicator = CommonExponentialMovingAverage({
            "window": window,
            "column": column,
            "output_column": self.name
        })

        # Create adapter
        self.adapter = IndicatorAdapter(self.indicator)

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EMA for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with EMA values
        """
        return self.adapter.calculate(data)


class WeightedMovingAverage:
    """
    Weighted Moving Average (WMA) indicator adapter.

    This class provides backward compatibility with the feature store service
    while using the common-lib implementation.
    """

    category = "moving_average"

    def __init__(self, window: int = 14, column: str = "close", **kwargs):
        """
        Initialize Weighted Moving Average indicator.

        Args:
            window: Lookback period for the moving average
            column: Data column to use for calculations (default: 'close')
            **kwargs: Additional parameters
        """
        self.window = window
        self.column = column
        self.name = f"wma_{window}"

        # Create common-lib indicator
        self.indicator = CommonWeightedMovingAverage({
            "window": window,
            "column": column,
            "output_column": self.name
        })

        # Create adapter
        self.adapter = IndicatorAdapter(self.indicator)

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate WMA for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with WMA values
        """
        return self.adapter.calculate(data)


class RelativeStrengthIndex:
    """
    Relative Strength Index (RSI) indicator adapter.

    This class provides backward compatibility with the feature store service
    while using the common-lib implementation.
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

        # Create common-lib indicator
        self.indicator = CommonRelativeStrengthIndex({
            "window": window,
            "column": column,
            "output_column": self.name
        })

        # Create adapter
        self.adapter = IndicatorAdapter(self.indicator)

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with RSI values
        """
        return self.adapter.calculate(data)


class Stochastic:
    """
    Stochastic Oscillator indicator adapter.

    This class provides backward compatibility with the feature store service
    while using the common-lib implementation.
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

        # Create common-lib indicator
        self.indicator = CommonStochastic({
            "k_window": k_window,
            "d_window": d_window,
            "d_method": d_method
        })

        # Create adapter
        self.adapter = IndicatorAdapter(self.indicator)

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Stochastic Oscillator values
        """
        return self.adapter.calculate(data)


class MACD:
    """
    Moving Average Convergence Divergence (MACD) indicator adapter.

    This class provides backward compatibility with the feature store service
    while using the common-lib implementation.
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

        # Create common-lib indicator
        self.indicator = CommonMACD({
            "fast_period": fast_period,
            "slow_period": slow_period,
            "signal_period": signal_period,
            "column": column
        })

        # Create adapter
        self.adapter = IndicatorAdapter(self.indicator)

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with MACD values
        """
        return self.adapter.calculate(data)


class CommodityChannelIndex:
    """
    Commodity Channel Index (CCI) indicator adapter.

    This class provides backward compatibility with the feature store service
    while using the common-lib implementation.
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

        # Create common-lib indicator
        self.indicator = CommonCommodityChannelIndex({
            "window": window,
            "constant": constant
        })

        # Create adapter
        self.adapter = IndicatorAdapter(self.indicator)

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate CCI for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with CCI values
        """
        return self.adapter.calculate(data)


class WilliamsR:
    """
    Williams %R indicator adapter.

    This class provides backward compatibility with the feature store service
    while using the common-lib implementation.
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

        # Create common-lib indicator
        self.indicator = CommonWilliamsR({
            "window": window
        })

        # Create adapter
        self.adapter = IndicatorAdapter(self.indicator)

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Williams %R for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Williams %R values
        """
        return self.adapter.calculate(data)


class RateOfChange:
    """
    Rate of Change (ROC) indicator adapter.

    This class provides backward compatibility with the feature store service
    while using the common-lib implementation.
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

        # Create common-lib indicator
        self.indicator = CommonRateOfChange({
            "window": window,
            "column": column,
            "method": method
        })

        # Create adapter
        self.adapter = IndicatorAdapter(self.indicator)

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ROC for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with ROC values
        """
        return self.adapter.calculate(data)


class BollingerBands:
    """
    Bollinger Bands indicator adapter.

    This class provides backward compatibility with the feature store service
    while using the common-lib implementation.
    """

    category = "volatility"

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        column: str = "close",
        **kwargs
    ):
        """
        Initialize Bollinger Bands indicator.

        Args:
            window: Lookback period for the moving average
            num_std: Number of standard deviations for the bands
            column: Data column to use for calculations (default: 'close')
            **kwargs: Additional parameters
        """
        self.window = window
        self.num_std = num_std
        self.column = column
        self.name_middle = f"bb_middle_{window}"
        self.name_upper = f"bb_upper_{window}_{num_std}"
        self.name_lower = f"bb_lower_{window}_{num_std}"
        self.name_width = f"bb_width_{window}_{num_std}"
        self.name_pct_b = f"bb_pct_b_{window}_{num_std}"

        # Create common-lib indicator
        self.indicator = CommonBollingerBands({
            "window": window,
            "num_std": num_std,
            "column": column
        })

        # Create adapter
        self.adapter = IndicatorAdapter(self.indicator)

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Bollinger Bands values
        """
        return self.adapter.calculate(data)


class KeltnerChannels:
    """
    Keltner Channels indicator adapter.

    This class provides backward compatibility with the feature store service
    while using the common-lib implementation.
    """

    category = "volatility"

    def __init__(
        self,
        window: int = 20,
        atr_window: int = 10,
        atr_multiplier: float = 2.0,
        ma_method: str = "ema",
        column: str = "close",
        **kwargs
    ):
        """
        Initialize Keltner Channels indicator.

        Args:
            window: Lookback period for the moving average
            atr_window: Lookback period for the ATR calculation
            atr_multiplier: Multiplier for the ATR
            ma_method: Moving average type ('sma' or 'ema')
            column: Data column to use for calculations (default: 'close')
            **kwargs: Additional parameters
        """
        self.window = window
        self.atr_window = atr_window
        self.atr_multiplier = atr_multiplier
        self.ma_method = ma_method
        self.column = column
        self.name_middle = f"kc_middle_{window}_{ma_method}"
        self.name_upper = f"kc_upper_{window}_{atr_window}_{atr_multiplier}"
        self.name_lower = f"kc_lower_{window}_{atr_window}_{atr_multiplier}"

        # Create common-lib indicator
        self.indicator = CommonKeltnerChannels({
            "window": window,
            "atr_window": atr_window,
            "atr_multiplier": atr_multiplier,
            "ma_method": ma_method,
            "column": column
        })

        # Create adapter
        self.adapter = IndicatorAdapter(self.indicator)

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Keltner Channels for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Keltner Channels values
        """
        return self.adapter.calculate(data)


class DonchianChannels:
    """
    Donchian Channels indicator adapter.

    This class provides backward compatibility with the feature store service
    while using the common-lib implementation.
    """

    category = "volatility"

    def __init__(
        self,
        window: int = 20,
        **kwargs
    ):
        """
        Initialize Donchian Channels indicator.

        Args:
            window: Lookback period for the channel calculation
            **kwargs: Additional parameters
        """
        self.window = window
        self.name_upper = f"donchian_upper_{window}"
        self.name_lower = f"donchian_lower_{window}"
        self.name_middle = f"donchian_middle_{window}"
        self.name_width = f"donchian_width_{window}"

        # Create common-lib indicator
        self.indicator = CommonDonchianChannels({
            "window": window
        })

        # Create adapter
        self.adapter = IndicatorAdapter(self.indicator)

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Donchian Channels for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Donchian Channels values
        """
        return self.adapter.calculate(data)


class AverageTrueRange:
    """
    Average True Range (ATR) indicator adapter.

    This class provides backward compatibility with the feature store service
    while using the common-lib implementation.
    """

    category = "volatility"

    def __init__(self, window: int = 14, **kwargs):
        """
        Initialize Average True Range indicator.

        Args:
            window: Lookback period for the ATR calculation
            **kwargs: Additional parameters
        """
        self.window = window
        self.name = f"atr_{window}"

        # Create common-lib indicator
        self.indicator = CommonAverageTrueRange({
            "window": window
        })

        # Create adapter
        self.adapter = IndicatorAdapter(self.indicator)

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ATR for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with ATR values
        """
        return self.adapter.calculate(data)


class PriceEnvelopes:
    """
    Price Envelopes indicator adapter.

    This class provides backward compatibility with the feature store service
    while using the common-lib implementation.
    """

    category = "volatility"

    def __init__(
        self,
        window: int = 20,
        percent: float = 2.5,
        ma_method: str = "sma",
        column: str = "close",
        **kwargs
    ):
        """
        Initialize Price Envelopes indicator.

        Args:
            window: Lookback period for the moving average
            percent: Percentage for envelope width
            ma_method: Moving average type ('sma' or 'ema')
            column: Data column to use for calculations (default: 'close')
            **kwargs: Additional parameters
        """
        self.window = window
        self.percent = percent
        self.ma_method = ma_method
        self.column = column
        self.name_middle = f"env_middle_{window}_{ma_method}"
        self.name_upper = f"env_upper_{window}_{percent}"
        self.name_lower = f"env_lower_{window}_{percent}"

        # Create common-lib indicator
        self.indicator = CommonPriceEnvelopes({
            "window": window,
            "percent": percent,
            "ma_method": ma_method,
            "column": column
        })

        # Create adapter
        self.adapter = IndicatorAdapter(self.indicator)

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Price Envelopes for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Price Envelopes values
        """
        return self.adapter.calculate(data)


class HistoricalVolatility:
    """
    Historical Volatility indicator adapter.

    This class provides backward compatibility with the feature store service
    while using the common-lib implementation.
    """

    category = "volatility"

    def __init__(
        self,
        window: int = 20,
        column: str = "close",
        annualize: bool = True,
        trading_periods: int = 252,
        **kwargs
    ):
        """
        Initialize Historical Volatility indicator.

        Args:
            window: Lookback period for calculation
            column: Data column to use for calculations (default: 'close')
            annualize: Whether to annualize the volatility
            trading_periods: Number of trading periods in a year
            **kwargs: Additional parameters
        """
        self.window = window
        self.column = column
        self.annualize = annualize
        self.trading_periods = trading_periods
        self.name = f"hv_{window}"

        # Create common-lib indicator
        self.indicator = CommonHistoricalVolatility({
            "window": window,
            "column": column,
            "annualize": annualize,
            "trading_periods": trading_periods
        })

        # Create adapter
        self.adapter = IndicatorAdapter(self.indicator)

    @log_and_time
    @cache_indicator(ttl=3600)  # Cache for 1 hour
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Historical Volatility for the given data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with Historical Volatility values
        """
        return self.adapter.calculate(data)