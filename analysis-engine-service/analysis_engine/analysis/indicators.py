"""
Technical Analysis Indicators Module

This module provides a comprehensive set of technical analysis indicators
and calculations used by various analyzers in the system.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import talib
import logging

logger = logging.getLogger(__name__)

@dataclass
class IndicatorResult:
    """Container for indicator calculation results"""
    name: str
    values: Union[np.ndarray, pd.Series]
    parameters: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "values": self.values.tolist() if isinstance(self.values, np.ndarray) else self.values.to_list(),
            "parameters": self.parameters,
            "metadata": self.metadata or {}
        }

class IndicatorClient:
    """
    Client for calculating technical analysis indicators

    Features:
    - Moving averages (SMA, EMA, WMA)
    - Momentum indicators (RSI, MACD, Stochastic)
    - Volatility indicators (Bollinger Bands, ATR)
    - Volume indicators (OBV, MFI)
    - Trend indicators (ADX, Aroon)
    """

    def __init__(self):
        """Initialize indicator client"""
        self.logger = logging.getLogger(f"{__name__}.IndicatorClient")

        # Initialize the common-lib adapter
        try:
            from analysis_engine.analysis.indicators.common_lib_adapter import CommonLibIndicatorAdapter
            self.common_lib_adapter = CommonLibIndicatorAdapter()
            self.use_common_lib = True
            self.logger.info("Using common-lib indicator implementations")
        except ImportError:
            self.use_common_lib = False
            self.logger.warning("Common-lib adapter not available, falling back to talib")

    def calculate_sma(
        self,
        data: Union[np.ndarray, pd.Series],
        period: int
    ) -> pd.Series:
        """
        Calculate Simple Moving Average

        Args:
            data: Price data
            period: Moving average period

        Returns:
            Series containing SMA values
        """
        try:
            # Use common-lib implementation if available
            if self.use_common_lib:
                return self.common_lib_adapter.calculate_sma(data, period)

            # Fall back to talib implementation
            return pd.Series(talib.SMA(data, timeperiod=period))
        except Exception as e:
            self.logger.error(f"Error calculating SMA: {str(e)}")
            return pd.Series(np.nan, index=range(len(data)))

    def calculate_ema(
        self,
        data: Union[np.ndarray, pd.Series],
        period: int
    ) -> pd.Series:
        """
        Calculate Exponential Moving Average

        Args:
            data: Price data
            period: Moving average period

        Returns:
            Series containing EMA values
        """
        try:
            # Use common-lib implementation if available
            if self.use_common_lib:
                return self.common_lib_adapter.calculate_ema(data, period)

            # Fall back to talib implementation
            return pd.Series(talib.EMA(data, timeperiod=period))
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {str(e)}")
            return pd.Series(np.nan, index=range(len(data)))

    def calculate_rsi(
        self,
        data: Union[np.ndarray, pd.Series],
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Relative Strength Index

        Args:
            data: Price data
            period: RSI period

        Returns:
            Series containing RSI values
        """
        try:
            # Use common-lib implementation if available
            if self.use_common_lib:
                return self.common_lib_adapter.calculate_rsi(data, period)

            # Fall back to talib implementation
            return pd.Series(talib.RSI(data, timeperiod=period))
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(np.nan, index=range(len(data)))

    def calculate_macd(
        self,
        data: Union[np.ndarray, pd.Series],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Args:
            data: Price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            Dictionary containing MACD line, signal line, and histogram
        """
        try:
            # Use common-lib implementation if available
            if self.use_common_lib:
                macd, signal, hist = self.common_lib_adapter.calculate_macd(
                    data, fast_period, slow_period, signal_period
                )
                return {
                    "macd": macd,
                    "signal": signal,
                    "histogram": hist
                }

            # Fall back to talib implementation
            macd, signal, hist = talib.MACD(
                data,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period
            )
            return {
                "macd": pd.Series(macd),
                "signal": pd.Series(signal),
                "histogram": pd.Series(hist)
            }
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {str(e)}")
            empty = pd.Series(np.nan, index=range(len(data)))
            return {
                "macd": empty,
                "signal": empty,
                "histogram": empty
            }

    def calculate_bollinger_bands(
        self,
        data: Union[np.ndarray, pd.Series],
        period: int = 20,
        num_std: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands

        Args:
            data: Price data
            period: Moving average period
            num_std: Number of standard deviations

        Returns:
            Dictionary containing upper band, middle band, and lower band
        """
        try:
            # Use common-lib implementation if available
            if self.use_common_lib:
                result = self.common_lib_adapter.calculate_bollinger_bands(data, period, num_std)
                # Return only the bands to maintain backward compatibility
                return {
                    "upper": result["upper"],
                    "middle": result["middle"],
                    "lower": result["lower"]
                }

            # Fall back to talib implementation
            upper, middle, lower = talib.BBANDS(
                data,
                timeperiod=period,
                nbdevup=num_std,
                nbdevdn=num_std
            )
            return {
                "upper": pd.Series(upper),
                "middle": pd.Series(middle),
                "lower": pd.Series(lower)
            }
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            empty = pd.Series(np.nan, index=range(len(data)))
            return {
                "upper": empty,
                "middle": empty,
                "lower": empty
            }

    def calculate_atr(
        self,
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            Series containing ATR values
        """
        try:
            # Use common-lib implementation if available
            if self.use_common_lib:
                # Prepare data for common-lib adapter
                if isinstance(high, np.ndarray) and isinstance(low, np.ndarray) and isinstance(close, np.ndarray):
                    # Create a DataFrame with OHLC data
                    data = pd.DataFrame({
                        "high": high,
                        "low": low,
                        "close": close
                    })
                else:
                    # Assume data is already in Series format
                    data = pd.DataFrame({
                        "high": high,
                        "low": low,
                        "close": close
                    })

                return self.common_lib_adapter.calculate_atr(data, period)

            # Fall back to talib implementation
            return pd.Series(talib.ATR(high, low, close, timeperiod=period))
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(np.nan, index=range(len(close)))

    def calculate_stochastic(
        self,
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        k_period: int = 14,
        d_period: int = 3,
        slowing: int = 3
    ) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period
            d_period: %D period
            slowing: Slowing period

        Returns:
            Dictionary containing %K and %D values
        """
        try:
            # Use common-lib implementation if available
            if self.use_common_lib:
                # Prepare data for common-lib adapter
                if isinstance(high, np.ndarray) and isinstance(low, np.ndarray) and isinstance(close, np.ndarray):
                    # Create a DataFrame with OHLC data
                    data = pd.DataFrame({
                        "high": high,
                        "low": low,
                        "close": close
                    })
                else:
                    # Assume data is already in Series format
                    data = pd.DataFrame({
                        "high": high,
                        "low": low,
                        "close": close
                    })

                k, d = self.common_lib_adapter.calculate_stochastic(
                    data, k_period, d_period, slowing
                )
                return {
                    "k": k,
                    "d": d
                }

            # Fall back to talib implementation
            k, d = talib.STOCH(
                high,
                low,
                close,
                fastk_period=k_period,
                slowk_period=slowing,
                slowk_matype=0,
                slowd_period=d_period,
                slowd_matype=0
            )
            return {
                "k": pd.Series(k),
                "d": pd.Series(d)
            }
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {str(e)}")
            empty = pd.Series(np.nan, index=range(len(close)))
            return {
                "k": empty,
                "d": empty
            }

    def calculate_adx(
        self,
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        period: int = 14
    ) -> Dict[str, pd.Series]:
        """
        Calculate Average Directional Index

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX period

        Returns:
            Dictionary containing ADX, +DI, and -DI values
        """
        try:
            adx = talib.ADX(high, low, close, timeperiod=period)
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=period)
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=period)
            return {
                "adx": pd.Series(adx),
                "plus_di": pd.Series(plus_di),
                "minus_di": pd.Series(minus_di)
            }
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {str(e)}")
            empty = pd.Series(np.nan, index=range(len(close)))
            return {
                "adx": empty,
                "plus_di": empty,
                "minus_di": empty
            }

    def calculate_obv(
        self,
        close: Union[np.ndarray, pd.Series],
        volume: Union[np.ndarray, pd.Series]
    ) -> pd.Series:
        """
        Calculate On Balance Volume

        Args:
            close: Close prices
            volume: Volume data

        Returns:
            Series containing OBV values
        """
        try:
            return pd.Series(talib.OBV(close, volume))
        except Exception as e:
            self.logger.error(f"Error calculating OBV: {str(e)}")
            return pd.Series(np.nan, index=range(len(close)))

    def calculate_mfi(
        self,
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        volume: Union[np.ndarray, pd.Series],
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Money Flow Index

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            period: MFI period

        Returns:
            Series containing MFI values
        """
        try:
            return pd.Series(talib.MFI(high, low, close, volume, timeperiod=period))
        except Exception as e:
            self.logger.error(f"Error calculating MFI: {str(e)}")
            return pd.Series(np.nan, index=range(len(close)))

    def calculate_aroon(
        self,
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        period: int = 25
    ) -> Dict[str, pd.Series]:
        """
        Calculate Aroon Indicator

        Args:
            high: High prices
            low: Low prices
            period: Aroon period

        Returns:
            Dictionary containing Aroon Up and Aroon Down values
        """
        try:
            aroon_down, aroon_up = talib.AROON(high, low, timeperiod=period)
            return {
                "aroon_up": pd.Series(aroon_up),
                "aroon_down": pd.Series(aroon_down)
            }
        except Exception as e:
            self.logger.error(f"Error calculating Aroon: {str(e)}")
            empty = pd.Series(np.nan, index=range(len(high)))
            return {
                "aroon_up": empty,
                "aroon_down": empty
            }

    def calculate_ichimoku(
        self,
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52
    ) -> Dict[str, pd.Series]:
        """
        Calculate Ichimoku Cloud components

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            tenkan_period: Tenkan-sen (Conversion Line) period
            kijun_period: Kijun-sen (Base Line) period
            senkou_b_period: Senkou Span B period

        Returns:
            Dictionary containing Ichimoku components
        """
        try:
            # Calculate Tenkan-sen (Conversion Line)
            high_tenkan = pd.Series(high).rolling(window=tenkan_period).max()
            low_tenkan = pd.Series(low).rolling(window=tenkan_period).min()
            tenkan_sen = (high_tenkan + low_tenkan) / 2

            # Calculate Kijun-sen (Base Line)
            high_kijun = pd.Series(high).rolling(window=kijun_period).max()
            low_kijun = pd.Series(low).rolling(window=kijun_period).min()
            kijun_sen = (high_kijun + low_kijun) / 2

            # Calculate Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)

            # Calculate Senkou Span B (Leading Span B)
            high_senkou = pd.Series(high).rolling(window=senkou_b_period).max()
            low_senkou = pd.Series(low).rolling(window=senkou_b_period).min()
            senkou_span_b = ((high_senkou + low_senkou) / 2).shift(kijun_period)

            # Calculate Chikou Span (Lagging Span)
            chikou_span = pd.Series(close).shift(-kijun_period)

            return {
                "tenkan_sen": tenkan_sen,
                "kijun_sen": kijun_sen,
                "senkou_span_a": senkou_span_a,
                "senkou_span_b": senkou_span_b,
                "chikou_span": chikou_span
            }
        except Exception as e:
            self.logger.error(f"Error calculating Ichimoku: {str(e)}")
            empty = pd.Series(np.nan, index=range(len(close)))
            return {
                "tenkan_sen": empty,
                "kijun_sen": empty,
                "senkou_span_a": empty,
                "senkou_span_b": empty,
                "chikou_span": empty
            }