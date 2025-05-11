"""
Common-Lib Indicator Adapter Module

This module provides adapters for using common-lib indicators in the analysis engine service.
It ensures backward compatibility with existing code while leveraging the standardized
indicator implementations from common-lib.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
import logging

from common_lib.indicators.base_indicator import BaseIndicator
from common_lib.indicators.moving_averages import (
    SimpleMovingAverage,
    ExponentialMovingAverage,
    WeightedMovingAverage
)
from common_lib.indicators.oscillators import (
    RelativeStrengthIndex,
    Stochastic,
    MACD,
    CommodityChannelIndex,
    WilliamsR,
    RateOfChange
)
from common_lib.indicators.volatility import (
    BollingerBands,
    KeltnerChannels,
    DonchianChannels,
    AverageTrueRange,
    PriceEnvelopes,
    HistoricalVolatility
)

logger = logging.getLogger(__name__)


class CommonLibIndicatorAdapter:
    """
    Adapter for common-lib indicators to be used with the IndicatorClient interface.

    This class provides a bridge between the common-lib indicator implementations
    and the analysis engine service's IndicatorClient interface.
    """

    def __init__(self):
        """Initialize the adapter."""
        self.logger = logging.getLogger(f"{__name__}.CommonLibIndicatorAdapter")
        self._indicators = {}

    def _get_indicator(self, indicator_class, **params):
        """
        Get or create an indicator instance.

        Args:
            indicator_class: Indicator class to instantiate
            **params: Parameters for the indicator

        Returns:
            Indicator instance
        """
        # Create a key for the indicator based on its class and parameters
        key = f"{indicator_class.__name__}_{str(sorted(params.items()))}"

        # Return existing instance if available
        if key in self._indicators:
            return self._indicators[key]

        # Create new instance
        try:
            indicator = indicator_class(params)
            self._indicators[key] = indicator
            return indicator
        except Exception as e:
            self.logger.error(f"Error creating indicator {indicator_class.__name__}: {str(e)}")
            raise

    def _prepare_data(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        column: str = "close",
        high_col: str = "high",
        low_col: str = "low"
    ) -> pd.DataFrame:
        """
        Prepare data for indicator calculation.

        Args:
            data: Price data
            column: Column name for price data
            high_col: Column name for high price data
            low_col: Column name for low price data

        Returns:
            DataFrame with prepared data
        """
        # Convert data to DataFrame if it's not already
        if isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                # Single array, assume it's close prices
                df = pd.DataFrame({column: data})
            elif len(data.shape) == 2 and data.shape[1] >= 3:
                # OHLC data
                df = pd.DataFrame({
                    "open": data[:, 0],
                    high_col: data[:, 1],
                    low_col: data[:, 2],
                    column: data[:, 3]
                })
            else:
                # Default to close prices
                df = pd.DataFrame({column: data.flatten()})
        elif isinstance(data, pd.Series):
            df = pd.DataFrame({column: data})
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        return df

    def calculate_sma(
        self,
        data: Union[np.ndarray, pd.Series],
        period: int
    ) -> pd.Series:
        """
        Calculate Simple Moving Average using common-lib implementation.

        Args:
            data: Price data
            period: Moving average period

        Returns:
            Series containing SMA values
        """
        try:
            # Prepare data
            df = self._prepare_data(data)

            # Get indicator instance
            indicator = self._get_indicator(
                SimpleMovingAverage,
                window=period,
                column="close",
                output_column=f"sma_{period}"
            )

            # Calculate indicator
            result = indicator.calculate(df)

            # Return the result as a Series
            return result[f"sma_{period}"]
        except Exception as e:
            self.logger.error(f"Error calculating SMA: {str(e)}")
            return pd.Series(np.nan, index=range(len(data)))

    def calculate_ema(
        self,
        data: Union[np.ndarray, pd.Series],
        period: int
    ) -> pd.Series:
        """
        Calculate Exponential Moving Average using common-lib implementation.

        Args:
            data: Price data
            period: Moving average period

        Returns:
            Series containing EMA values
        """
        try:
            # Prepare data
            df = self._prepare_data(data)

            # Get indicator instance
            indicator = self._get_indicator(
                ExponentialMovingAverage,
                window=period,
                column="close",
                output_column=f"ema_{period}"
            )

            # Calculate indicator
            result = indicator.calculate(df)

            # Return the result as a Series
            return result[f"ema_{period}"]
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {str(e)}")
            return pd.Series(np.nan, index=range(len(data)))

    def calculate_wma(
        self,
        data: Union[np.ndarray, pd.Series],
        period: int
    ) -> pd.Series:
        """
        Calculate Weighted Moving Average using common-lib implementation.

        Args:
            data: Price data
            period: Moving average period

        Returns:
            Series containing WMA values
        """
        try:
            # Prepare data
            df = self._prepare_data(data)

            # Get indicator instance
            indicator = self._get_indicator(
                WeightedMovingAverage,
                window=period,
                column="close",
                output_column=f"wma_{period}"
            )

            # Calculate indicator
            result = indicator.calculate(df)

            # Return the result as a Series
            return result[f"wma_{period}"]
        except Exception as e:
            self.logger.error(f"Error calculating WMA: {str(e)}")
            return pd.Series(np.nan, index=range(len(data)))

    def calculate_rsi(
        self,
        data: Union[np.ndarray, pd.Series],
        period: int
    ) -> pd.Series:
        """
        Calculate Relative Strength Index using common-lib implementation.

        Args:
            data: Price data
            period: RSI period

        Returns:
            Series containing RSI values
        """
        try:
            # Prepare data
            df = self._prepare_data(data)

            # Get indicator instance
            indicator = self._get_indicator(
                RelativeStrengthIndex,
                window=period,
                column="close",
                output_column=f"rsi_{period}"
            )

            # Calculate indicator
            result = indicator.calculate(df)

            # Return the result as a Series
            return result[f"rsi_{period}"]
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(np.nan, index=range(len(data)))

    def calculate_stochastic(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        k_period: int = 14,
        d_period: int = 3,
        slowing: int = 1
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator using common-lib implementation.

        Args:
            data: Price data (OHLC)
            k_period: %K period
            d_period: %D period
            slowing: Slowing period (not used in common-lib implementation)

        Returns:
            Tuple of Series containing %K and %D values
        """
        try:
            # Prepare data
            df = self._prepare_data(data)

            # Get indicator instance
            indicator = self._get_indicator(
                Stochastic,
                k_window=k_period,
                d_window=d_period,
                d_method="sma"
            )

            # Calculate indicator
            result = indicator.calculate(df)

            # Return the result as Series
            k_series = result[f"stoch_k_{k_period}"]
            d_series = result[f"stoch_d_{k_period}_{d_period}"]

            return k_series, d_series
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {str(e)}")
            empty_series = pd.Series(np.nan, index=range(len(data)))
            return empty_series, empty_series

    def calculate_macd(
        self,
        data: Union[np.ndarray, pd.Series],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD using common-lib implementation.

        Args:
            data: Price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal EMA period

        Returns:
            Tuple of Series containing MACD line, signal line, and histogram values
        """
        try:
            # Prepare data
            df = self._prepare_data(data)

            # Get indicator instance
            indicator = self._get_indicator(
                MACD,
                fast_period=fast_period,
                slow_period=slow_period,
                signal_period=signal_period,
                column="close"
            )

            # Calculate indicator
            result = indicator.calculate(df)

            # Return the result as Series
            macd_line = result[f"macd_{fast_period}_{slow_period}"]
            signal_line = result[f"macd_signal_{signal_period}"]
            histogram = result[f"macd_hist_{fast_period}_{slow_period}_{signal_period}"]

            return macd_line, signal_line, histogram
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {str(e)}")
            empty_series = pd.Series(np.nan, index=range(len(data)))
            return empty_series, empty_series, empty_series

    def calculate_cci(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        period: int = 20
    ) -> pd.Series:
        """
        Calculate Commodity Channel Index using common-lib implementation.

        Args:
            data: Price data (OHLC)
            period: CCI period

        Returns:
            Series containing CCI values
        """
        try:
            # Prepare data
            df = self._prepare_data(data)

            # Get indicator instance
            indicator = self._get_indicator(
                CommodityChannelIndex,
                window=period,
                constant=0.015
            )

            # Calculate indicator
            result = indicator.calculate(df)

            # Return the result as a Series
            return result[f"cci_{period}"]
        except Exception as e:
            self.logger.error(f"Error calculating CCI: {str(e)}")
            return pd.Series(np.nan, index=range(len(data)))

    def calculate_williams_r(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Williams %R using common-lib implementation.

        Args:
            data: Price data (OHLC)
            period: Lookback period

        Returns:
            Series containing Williams %R values
        """
        try:
            # Prepare data
            df = self._prepare_data(data)

            # Get indicator instance
            indicator = self._get_indicator(
                WilliamsR,
                window=period
            )

            # Calculate indicator
            result = indicator.calculate(df)

            # Return the result as a Series
            return result[f"williams_r_{period}"]
        except Exception as e:
            self.logger.error(f"Error calculating Williams %R: {str(e)}")
            return pd.Series(np.nan, index=range(len(data)))

    def calculate_roc(
        self,
        data: Union[np.ndarray, pd.Series],
        period: int = 10
    ) -> pd.Series:
        """
        Calculate Rate of Change using common-lib implementation.

        Args:
            data: Price data
            period: ROC period

        Returns:
            Series containing ROC values
        """
        try:
            # Prepare data
            df = self._prepare_data(data)

            # Get indicator instance
            indicator = self._get_indicator(
                RateOfChange,
                window=period,
                column="close",
                method="percentage"
            )

            # Calculate indicator
            result = indicator.calculate(df)

            # Return the result as a Series
            return result[f"roc_{period}"]
        except Exception as e:
            self.logger.error(f"Error calculating ROC: {str(e)}")
            return pd.Series(np.nan, index=range(len(data)))

    def calculate_bollinger_bands(
        self,
        data: Union[np.ndarray, pd.Series],
        period: int = 20,
        num_std: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands using common-lib implementation.

        Args:
            data: Price data
            period: Lookback period
            num_std: Number of standard deviations

        Returns:
            Dictionary containing upper, middle, lower bands, bandwidth, and %B
        """
        try:
            # Prepare data
            df = self._prepare_data(data)

            # Get indicator instance
            indicator = self._get_indicator(
                BollingerBands,
                window=period,
                num_std=num_std,
                column="close"
            )

            # Calculate indicator
            result = indicator.calculate(df)

            # Return the result as a dictionary of Series
            return {
                "upper": result[f"bb_upper_{period}_{num_std}"],
                "middle": result[f"bb_middle_{period}"],
                "lower": result[f"bb_lower_{period}_{num_std}"],
                "bandwidth": result[f"bb_width_{period}_{num_std}"],
                "percent_b": result[f"bb_pct_b_{period}_{num_std}"]
            }
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            empty_series = pd.Series(np.nan, index=range(len(data)))
            return {
                "upper": empty_series.copy(),
                "middle": empty_series.copy(),
                "lower": empty_series.copy(),
                "bandwidth": empty_series.copy(),
                "percent_b": empty_series.copy()
            }

    def calculate_keltner_channels(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        period: int = 20,
        atr_period: int = 10,
        atr_multiplier: float = 2.0,
        ma_method: str = "ema"
    ) -> Dict[str, pd.Series]:
        """
        Calculate Keltner Channels using common-lib implementation.

        Args:
            data: Price data (OHLC)
            period: Lookback period for the moving average
            atr_period: Lookback period for the ATR
            atr_multiplier: Multiplier for the ATR
            ma_method: Moving average type ('sma' or 'ema')

        Returns:
            Dictionary containing upper, middle, and lower bands
        """
        try:
            # Prepare data
            df = self._prepare_data(data)

            # Get indicator instance
            indicator = self._get_indicator(
                KeltnerChannels,
                window=period,
                atr_window=atr_period,
                atr_multiplier=atr_multiplier,
                ma_method=ma_method,
                column="close"
            )

            # Calculate indicator
            result = indicator.calculate(df)

            # Return the result as a dictionary of Series
            return {
                "upper": result[f"kc_upper_{period}_{atr_period}_{atr_multiplier}"],
                "middle": result[f"kc_middle_{period}_{ma_method}"],
                "lower": result[f"kc_lower_{period}_{atr_period}_{atr_multiplier}"]
            }
        except Exception as e:
            self.logger.error(f"Error calculating Keltner Channels: {str(e)}")
            empty_series = pd.Series(np.nan, index=range(len(data)))
            return {
                "upper": empty_series.copy(),
                "middle": empty_series.copy(),
                "lower": empty_series.copy()
            }

    def calculate_donchian_channels(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        period: int = 20
    ) -> Dict[str, pd.Series]:
        """
        Calculate Donchian Channels using common-lib implementation.

        Args:
            data: Price data (OHLC)
            period: Lookback period

        Returns:
            Dictionary containing upper, middle, lower bands, and width
        """
        try:
            # Prepare data
            df = self._prepare_data(data)

            # Get indicator instance
            indicator = self._get_indicator(
                DonchianChannels,
                window=period
            )

            # Calculate indicator
            result = indicator.calculate(df)

            # Return the result as a dictionary of Series
            return {
                "upper": result[f"donchian_upper_{period}"],
                "middle": result[f"donchian_middle_{period}"],
                "lower": result[f"donchian_lower_{period}"],
                "width": result[f"donchian_width_{period}"]
            }
        except Exception as e:
            self.logger.error(f"Error calculating Donchian Channels: {str(e)}")
            empty_series = pd.Series(np.nan, index=range(len(data)))
            return {
                "upper": empty_series.copy(),
                "middle": empty_series.copy(),
                "lower": empty_series.copy(),
                "width": empty_series.copy()
            }

    def calculate_atr(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range using common-lib implementation.

        Args:
            data: Price data (OHLC)
            period: Lookback period

        Returns:
            Series containing ATR values
        """
        try:
            # Prepare data
            df = self._prepare_data(data)

            # Get indicator instance
            indicator = self._get_indicator(
                AverageTrueRange,
                window=period
            )

            # Calculate indicator
            result = indicator.calculate(df)

            # Return the result as a Series
            return result[f"atr_{period}"]
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(np.nan, index=range(len(data)))

    def calculate_price_envelopes(
        self,
        data: Union[np.ndarray, pd.Series],
        period: int = 20,
        percent: float = 2.5,
        ma_method: str = "sma"
    ) -> Dict[str, pd.Series]:
        """
        Calculate Price Envelopes using common-lib implementation.

        Args:
            data: Price data
            period: Lookback period
            percent: Percentage for envelope width
            ma_method: Moving average type ('sma' or 'ema')

        Returns:
            Dictionary containing upper, middle, and lower bands
        """
        try:
            # Prepare data
            df = self._prepare_data(data)

            # Get indicator instance
            indicator = self._get_indicator(
                PriceEnvelopes,
                window=period,
                percent=percent,
                ma_method=ma_method,
                column="close"
            )

            # Calculate indicator
            result = indicator.calculate(df)

            # Return the result as a dictionary of Series
            return {
                "upper": result[f"env_upper_{period}_{percent}"],
                "middle": result[f"env_middle_{period}_{ma_method}"],
                "lower": result[f"env_lower_{period}_{percent}"]
            }
        except Exception as e:
            self.logger.error(f"Error calculating Price Envelopes: {str(e)}")
            empty_series = pd.Series(np.nan, index=range(len(data)))
            return {
                "upper": empty_series.copy(),
                "middle": empty_series.copy(),
                "lower": empty_series.copy()
            }

    def calculate_historical_volatility(
        self,
        data: Union[np.ndarray, pd.Series],
        period: int = 20,
        annualize: bool = True,
        trading_periods: int = 252
    ) -> pd.Series:
        """
        Calculate Historical Volatility using common-lib implementation.

        Args:
            data: Price data
            period: Lookback period
            annualize: Whether to annualize the volatility
            trading_periods: Number of trading periods in a year

        Returns:
            Series containing Historical Volatility values
        """
        try:
            # Prepare data
            df = self._prepare_data(data)

            # Get indicator instance
            indicator = self._get_indicator(
                HistoricalVolatility,
                window=period,
                column="close",
                annualize=annualize,
                trading_periods=trading_periods
            )

            # Calculate indicator
            result = indicator.calculate(df)

            # Return the result as a Series
            return result[f"hv_{period}"]
        except Exception as e:
            self.logger.error(f"Error calculating Historical Volatility: {str(e)}")
            return pd.Series(np.nan, index=range(len(data)))