"""
Timeframe Confluence Indicator implementation.

This module implements a system to measure indicator concordance
across multiple timeframes with visualization capabilities.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
from enum import Enum

from core.base_indicator import BaseIndicator


class SignalType(Enum):
    """Enum for signal types."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class TimeframeConfluenceIndicator(BaseIndicator):
    """
    Timeframe Confluence Indicator.

    This indicator measures the concordance of signals across multiple timeframes
    for any technical indicator or combination of indicators.
    Optimized for performance with caching and vectorized operations.
    """

    category = "multi_timeframe_concordance"

    # Class-level cache for signal calculations
    _signal_cache = {}
    _cache_timestamps = {}
    _cache_max_size = 100

    def __init__(
        self,
        signal_functions: Dict[str, Callable],
        timeframes: List[str],
        reference_timeframe: Optional[str] = None,
        concordance_window: int = 1,
        enable_caching: bool = True,
        cache_ttl: int = 300,  # 5 minutes
        use_vectorized_operations: bool = True,
        **kwargs
    ):
        """
        Initialize Timeframe Confluence Indicator with optimized performance.

        Args:
            signal_functions: Dictionary of signal functions, each returning 1 (bullish),
                             -1 (bearish), or 0 (neutral) for a given dataframe
            timeframes: List of timeframes to analyze
            reference_timeframe: Timeframe to align results to (defaults to lowest timeframe)
            concordance_window: Number of bars to check for confluence
            enable_caching: If True, cache signal calculations for improved performance
            cache_ttl: Time-to-live for cached results in seconds
            use_vectorized_operations: If True, use vectorized operations for better performance
            **kwargs: Additional parameters
        """
        self.signal_functions = signal_functions
        self.timeframes = timeframes
        self.reference_timeframe = reference_timeframe if reference_timeframe else timeframes[0]
        self.concordance_window = concordance_window
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.use_vectorized_operations = use_vectorized_operations
        self.kwargs = kwargs

        self.name = "timeframe_confluence"

    def calculate(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate Timeframe Confluence for the given data with optimized performance.

        Args:
            data: Dictionary of DataFrames with OHLCV data for each timeframe

        Returns:
            DataFrame with Timeframe Confluence indicator values
        """
        # Check if all required timeframes are available
        for tf in self.timeframes:
            if tf not in data:
                raise ValueError(f"Data must contain '{tf}' timeframe data")

        # Use reference timeframe as the base result
        result = data[self.reference_timeframe].copy()

        # Calculate signals for each indicator and timeframe
        signals = {}

        # Process each indicator
        for indicator_name, signal_func in self.signal_functions.items():
            signals[indicator_name] = {}

            # Process each timeframe
            for tf in self.timeframes:
                tf_data = data[tf].copy()

                # Calculate the signal (-1, 0, or 1)
                signal_col = f"{indicator_name}_{tf}_signal"

                # Check cache first if enabled
                cache_key = self._get_cache_key(indicator_name, tf, tf_data)
                cached_signal = self._get_from_cache(cache_key)

                if cached_signal is not None:
                    result[signal_col] = cached_signal
                else:
                    # Calculate the signal
                    signal = signal_func(tf_data)
                    result[signal_col] = signal

                    # Cache the result if enabled
                    if self.enable_caching:
                        self._add_to_cache(cache_key, signal)

                signals[indicator_name][tf] = signal_col

        # Calculate concordance for each indicator
        if self.use_vectorized_operations:
            # Vectorized calculation for all indicators at once
            self._calculate_all_concordance_vectorized(result, signals)
        else:
            # Calculate concordance for each indicator separately
            for indicator_name, tf_signals in signals.items():
                self._calculate_indicator_concordance(result, indicator_name, tf_signals)

            # Calculate overall concordance across all indicators
            self._calculate_overall_concordance(result, signals)

        return result

    def _calculate_all_concordance_vectorized(
        self,
        result: pd.DataFrame,
        signals: Dict[str, Dict[str, str]]
    ) -> None:
        """
        Calculate concordance for all indicators using vectorized operations.

        Args:
            result: DataFrame to store concordance
            signals: Nested dictionary of indicator names to timeframes to signal columns
        """
        # Process each indicator
        indicator_concordance_cols = []
        signal_type_cols = []

        for indicator_name, tf_signals in signals.items():
            # Create signal matrices for vectorized operations
            signal_cols = list(tf_signals.values())
            signal_matrix = result[signal_cols].values

            # Count signals using vectorized operations
            bullish_count = np.sum(signal_matrix > 0, axis=1)
            bearish_count = np.sum(signal_matrix < 0, axis=1)
            neutral_count = np.sum(signal_matrix == 0, axis=1)

            # Total number of timeframes
            num_timeframes = len(tf_signals)

            # Calculate concordance percentages
            result[f"{indicator_name}_bullish_concordance"] = bullish_count * 100 / num_timeframes
            result[f"{indicator_name}_bearish_concordance"] = bearish_count * 100 / num_timeframes
            result[f"{indicator_name}_neutral_concordance"] = neutral_count * 100 / num_timeframes

            # Determine signal type using vectorized operations
            signal_type = np.zeros(len(result))
            signal_type[bullish_count > bearish_count] = 1
            signal_type[bearish_count > bullish_count] = -1
            result[f"{indicator_name}_signal_type"] = signal_type

            # Calculate overall concordance for this indicator
            concordance_matrix = np.column_stack([
                result[f"{indicator_name}_bullish_concordance"].values,
                result[f"{indicator_name}_bearish_concordance"].values,
                result[f"{indicator_name}_neutral_concordance"].values
            ])
            result[f"{indicator_name}_concordance"] = np.max(concordance_matrix, axis=1)

            # Track columns for overall calculations
            indicator_concordance_cols.append(f"{indicator_name}_concordance")
            signal_type_cols.append(f"{indicator_name}_signal_type")

        # Calculate overall concordance across all indicators
        result["overall_concordance"] = result[indicator_concordance_cols].mean(axis=1)

        # Count bullish and bearish signals across all indicators
        signal_type_matrix = result[signal_type_cols].values
        bullish_count = np.sum(signal_type_matrix > 0, axis=1)
        bearish_count = np.sum(signal_type_matrix < 0, axis=1)

        # Calculate overall signal concordance
        total_indicators = len(signal_type_cols)
        result["overall_bullish_agreement"] = bullish_count * 100 / total_indicators
        result["overall_bearish_agreement"] = bearish_count * 100 / total_indicators

        # Overall signal direction
        overall_signal = np.zeros(len(result))
        overall_signal[bullish_count > bearish_count] = 1
        overall_signal[bearish_count > bullish_count] = -1
        result["overall_signal"] = overall_signal

    def _calculate_indicator_concordance(
        self,
        result: pd.DataFrame,
        indicator_name: str,
        tf_signals: Dict[str, str]
    ) -> None:
        """
        Calculate concordance for a specific indicator across timeframes.

        Args:
            result: DataFrame to store concordance
            indicator_name: Name of the indicator
            tf_signals: Dictionary mapping timeframes to signal column names
        """
        # Count bullish, bearish, and neutral signals
        bullish_signals = pd.DataFrame(index=result.index)
        bearish_signals = pd.DataFrame(index=result.index)
        neutral_signals = pd.DataFrame(index=result.index)

        for tf, col in tf_signals.items():
            bullish_signals[tf] = (result[col] > 0).astype(int)
            bearish_signals[tf] = (result[col] < 0).astype(int)
            neutral_signals[tf] = (result[col] == 0).astype(int)

        # Calculate concordance percentage for each signal type
        result[f"{indicator_name}_bullish_concordance"] = bullish_signals.sum(axis=1) * 100 / len(tf_signals)
        result[f"{indicator_name}_bearish_concordance"] = bearish_signals.sum(axis=1) * 100 / len(tf_signals)
        result[f"{indicator_name}_neutral_concordance"] = neutral_signals.sum(axis=1) * 100 / len(tf_signals)

        # Determine the dominant signal type
        result[f"{indicator_name}_signal_type"] = 0
        result.loc[result[f"{indicator_name}_bullish_concordance"] >
                 result[f"{indicator_name}_bearish_concordance"], f"{indicator_name}_signal_type"] = 1
        result.loc[result[f"{indicator_name}_bearish_concordance"] >
                 result[f"{indicator_name}_bullish_concordance"], f"{indicator_name}_signal_type"] = -1

        # Calculate overall concordance for this indicator (0-100%)
        result[f"{indicator_name}_concordance"] = result[[
            f"{indicator_name}_bullish_concordance",
            f"{indicator_name}_bearish_concordance",
            f"{indicator_name}_neutral_concordance"
        ]].max(axis=1)

    def _calculate_overall_concordance(
        self,
        result: pd.DataFrame,
        signals: Dict[str, Dict[str, str]]
    ) -> None:
        """
        Calculate overall concordance across all indicators and timeframes.

        Args:
            result: DataFrame to store concordance
            signals: Nested dictionary of indicator names to timeframes to signal columns
        """
        # Get all individual indicator concordance columns
        concordance_cols = [col for col in result.columns if col.endswith("_concordance")
                           and not col.endswith(("_bullish_concordance", "_bearish_concordance", "_neutral_concordance"))]

        # Calculate the mean concordance across all indicators
        result["overall_concordance"] = result[concordance_cols].mean(axis=1)

        # Get all signal type columns
        signal_type_cols = [col for col in result.columns if col.endswith("_signal_type")]

        # Count number of bullish and bearish signals
        bullish_count = (result[signal_type_cols] > 0).sum(axis=1)
        bearish_count = (result[signal_type_cols] < 0).sum(axis=1)

        # Calculate overall signal concordance
        total_signals = len(signal_type_cols)
        result["overall_bullish_agreement"] = bullish_count * 100 / total_signals
        result["overall_bearish_agreement"] = bearish_count * 100 / total_signals

        # Overall signal direction
        result["overall_signal"] = 0
        result.loc[bullish_count > bearish_count, "overall_signal"] = 1
        result.loc[bearish_count > bullish_count, "overall_signal"] = -1

    def _get_cache_key(self, indicator_name: str, timeframe: str, data: pd.DataFrame) -> str:
        """
        Generate a cache key for the given indicator, timeframe, and data.

        Args:
            indicator_name: Name of the indicator
            timeframe: Timeframe
            data: Input data

        Returns:
            Cache key string
        """
        # Use the indicator name, timeframe, and data shape/range as the key
        if len(data) > 0:
            start_time = data.index[0].isoformat()
            end_time = data.index[-1].isoformat()
            data_shape = f"{len(data)}_{len(data.columns)}"
        else:
            start_time = "empty"
            end_time = "empty"
            data_shape = "0_0"

        # Include indicator parameters in the key
        params_str = "_".join(f"{k}={v}" for k, v in self.kwargs.items())

        return f"{indicator_name}_{timeframe}_{data_shape}_{start_time}_{end_time}_{params_str}"

    def _get_from_cache(self, cache_key: str) -> Optional[pd.Series]:
        """
        Get a signal from the cache if available and not expired.

        Args:
            cache_key: Cache key

        Returns:
            Cached signal if available, None otherwise
        """
        if not self.enable_caching:
            return None

        if cache_key in self._signal_cache:
            # Check if the cache entry has expired
            timestamp = self._cache_timestamps.get(cache_key)
            if timestamp:
                current_time = pd.Timestamp.now()
                if (current_time - timestamp).total_seconds() < self.cache_ttl:
                    # Cache is still valid
                    return self._signal_cache[cache_key]

        return None

    def _add_to_cache(self, cache_key: str, signal: pd.Series) -> None:
        """
        Add a signal to the cache.

        Args:
            cache_key: Cache key
            signal: Signal to cache
        """
        if not self.enable_caching:
            return

        # Add to cache
        self._signal_cache[cache_key] = signal
        self._cache_timestamps[cache_key] = pd.Timestamp.now()

        # Clean up cache if it's too large
        if len(self._signal_cache) > self._cache_max_size:
            self._cleanup_cache()

    def _cleanup_cache(self) -> None:
        """Clean up the cache by removing the oldest entries."""
        # Sort cache keys by timestamp (oldest first)
        sorted_keys = sorted(
            self._cache_timestamps.keys(),
            key=lambda k: self._cache_timestamps[k]
        )

        # Remove the oldest half of the entries
        keys_to_remove = sorted_keys[:len(sorted_keys) // 2]

        for key in keys_to_remove:
            if key in self._signal_cache:
                del self._signal_cache[key]
            if key in self._cache_timestamps:
                del self._cache_timestamps[key]

    def get_visual_analysis_data(self, result: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate data for advanced visual analysis of concordance.

        Args:
            result: DataFrame with calculated concordance

        Returns:
            Dictionary with visualization data
        """
        indicators = list(self.signal_functions.keys())

        visualization_data = {
            "timeframes": self.timeframes,
            "indicators": indicators,
            "overall_concordance": result["overall_concordance"].tolist(),
            "overall_signal": result["overall_signal"].tolist(),
            "indicator_data": {}
        }

        for indicator in indicators:
            visualization_data["indicator_data"][indicator] = {
                "concordance": result[f"{indicator}_concordance"].tolist(),
                "signal_type": result[f"{indicator}_signal_type"].tolist(),
                "bullish_concordance": result[f"{indicator}_bullish_concordance"].tolist(),
                "bearish_concordance": result[f"{indicator}_bearish_concordance"].tolist()
            }

        return visualization_data


# Example signal functions
def rsi_signal(data: pd.DataFrame, column: str = "rsi_14", overbought: float = 70, oversold: float = 30) -> pd.Series:
    """Generate RSI signals: 1 for bullish (oversold), -1 for bearish (overbought), 0 for neutral."""
    signal = pd.Series(0, index=data.index)
    signal[data[column] < oversold] = 1
    signal[data[column] > overbought] = -1
    return signal

def macd_signal(data: pd.DataFrame, line_col: str = "macd_line", signal_col: str = "macd_signal") -> pd.Series:
    """Generate MACD signals: 1 for bullish (line crosses above signal), -1 for bearish (line crosses below signal)."""
    signal = pd.Series(0, index=data.index)
    signal[data[line_col] > data[signal_col]] = 1
    signal[data[line_col] < data[signal_col]] = -1
    return signal

def ma_cross_signal(data: pd.DataFrame, fast_ma: str = "sma_20", slow_ma: str = "sma_50") -> pd.Series:
    """Generate MA crossover signals: 1 for bullish (fast MA above slow MA), -1 for bearish (fast MA below slow MA)."""
    signal = pd.Series(0, index=data.index)
    signal[data[fast_ma] > data[slow_ma]] = 1
    signal[data[fast_ma] < data[slow_ma]] = -1
    return signal
