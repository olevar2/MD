"""
Multi-Timeframe Analysis Module.

This module provides tools for analyzing indicators across multiple timeframes
and integrating those signals to form more robust trading strategies.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
from enum import Enum
import warnings

from feature_store_service.indicators.base_indicator import BaseIndicator


class TimeframeAlignment(Enum):
    """Enum for timeframe alignment methods."""
    UPSAMPLE = "upsample"  # Higher timeframes values repeated on lower timeframe bars
    DOWNSAMPLE = "downsample"  # Lower timeframe values aggregated to higher timeframes
    HYBRID = "hybrid"  # Mix of upsampling and downsampling based on reference timeframe


class TimeframeRelation(Enum):
    """Enum for describing the relationship between the indicator's timeframe and the reference."""
    LOWER = "lower"  # Indicator timeframe is lower than reference (e.g., 1H vs 4H)
    SAME = "same"  # Indicator timeframe is the same as reference
    HIGHER = "higher"  # Indicator timeframe is higher than reference (e.g., 4H vs 1H)


class MultiTimeframeIndicator(BaseIndicator):
    """
    Multi-Timeframe Indicator

    This indicator wraps an existing indicator and allows it to be calculated
    across multiple timeframes, with the results aligned to the input timeframe.
    Optimized for performance with hierarchical computation and caching.
    """

    category = "multi_timeframe"

    # Class-level cache for indicator results
    _result_cache = {}
    _cache_timestamps = {}
    _cache_max_size = 100

    def __init__(
        self,
        base_indicator: BaseIndicator,
        timeframes: List[str],
        reference_timeframe: Optional[str] = None,
        alignment_method: str = "upsample",
        use_hierarchical_computation: bool = True,
        enable_caching: bool = True,
        cache_ttl: int = 300,  # 5 minutes
        **kwargs
    ):
        """
        Initialize Multi-Timeframe Indicator with optimized performance.

        Args:
            base_indicator: Base indicator instance to apply across timeframes
            timeframes: List of timeframes to analyze (e.g., ["1m", "5m", "15m", "1h"])
            reference_timeframe: Timeframe to align results to (default: lowest timeframe)
            alignment_method: Method for aligning timeframes ("upsample", "downsample", "hybrid")
            use_hierarchical_computation: If True, use hierarchical computation for efficiency
            enable_caching: If True, cache results for improved performance
            cache_ttl: Time-to-live for cached results in seconds
            **kwargs: Additional parameters passed to the base indicator
        """
        self.base_indicator = base_indicator
        self.timeframes = timeframes
        self.reference_timeframe = reference_timeframe or self._get_lowest_timeframe(timeframes)
        self.alignment_method = alignment_method
        self.use_hierarchical_computation = use_hierarchical_computation
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl

        # Additional parameters for the base indicator
        self.kwargs = kwargs

        # Sort timeframes by size for hierarchical computation
        self.sorted_timeframes = self._sort_timeframes_by_size(timeframes)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the indicator across multiple timeframes and align results.
        Optimized with hierarchical computation and caching.

        Args:
            data: DataFrame with OHLCV data at the reference timeframe

        Returns:
            DataFrame with indicator values for all specified timeframes
        """
        if 'timestamp' not in data.columns:
            raise ValueError("Data must contain 'timestamp' column for timeframe conversion")

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Ensure timestamp is a datetime index or column
        if not pd.api.types.is_datetime64_any_dtype(result['timestamp']):
            result['timestamp'] = pd.to_datetime(result['timestamp'])

        # Set timestamp as index for easier resampling
        result.set_index('timestamp', inplace=True)

        # Check if we should use hierarchical computation
        if self.use_hierarchical_computation:
            # Process timeframes in order from largest to smallest for hierarchical computation
            result = self._calculate_hierarchical(result)
        else:
            # Process each timeframe independently
            result = self._calculate_independent(result)

        # Restore the timestamp as a column
        result.reset_index(inplace=True)

        return result

    def _calculate_hierarchical(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicators using hierarchical computation for efficiency.
        Process larger timeframes first, then derive smaller timeframes where possible.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with indicator values for all timeframes
        """
        result = data.copy()

        # Process timeframes from largest to smallest
        processed_results = {}

        for tf in self.sorted_timeframes:
            # Check if we can derive this timeframe from a larger one
            parent_tf = self._find_parent_timeframe(tf, processed_results.keys())

            if parent_tf:
                # Try to derive from parent timeframe
                tf_result = self._derive_from_parent(result, tf, parent_tf, processed_results[parent_tf])
            else:
                # Calculate directly
                tf_relation = self._get_timeframe_relation(tf, self.reference_timeframe)

                if tf_relation == TimeframeRelation.SAME:
                    # Check cache first
                    cache_key = self._get_cache_key(data, tf)
                    cached_result = self._get_from_cache(cache_key)

                    if cached_result is not None:
                        tf_result = cached_result
                    else:
                        # No conversion needed, calculate directly
                        tf_result = self._calculate_for_timeframe(result, tf)

                        # Cache the result
                        if self.enable_caching:
                            self._add_to_cache(cache_key, tf_result)
                elif tf_relation == TimeframeRelation.HIGHER:
                    # Need to downsample the data
                    downsampled_data = self._downsample_data(result, tf)

                    # Check cache first
                    cache_key = self._get_cache_key(downsampled_data, tf)
                    cached_result = self._get_from_cache(cache_key)

                    if cached_result is not None:
                        tf_result = cached_result
                    else:
                        # Calculate indicator
                        tf_result = self._calculate_for_timeframe(downsampled_data, tf)

                        # Cache the result
                        if self.enable_caching:
                            self._add_to_cache(cache_key, tf_result)
                else:  # TimeframeRelation.LOWER
                    # Need to upsample the data
                    upsampled_data = self._upsample_data(result, tf)

                    # Check cache first
                    cache_key = self._get_cache_key(upsampled_data, tf)
                    cached_result = self._get_from_cache(cache_key)

                    if cached_result is not None:
                        tf_result = cached_result
                    else:
                        # Calculate indicator
                        tf_result = self._calculate_for_timeframe(upsampled_data, tf)

                        # Cache the result
                        if self.enable_caching:
                            self._add_to_cache(cache_key, tf_result)

            # Store the result for potential use by smaller timeframes
            processed_results[tf] = tf_result

            # Merge results into the output DataFrame
            self._merge_results(result, tf, tf_result)

        return result

    def _calculate_independent(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicators for each timeframe independently.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with indicator values for all timeframes
        """
        result = data.copy()

        # For each timeframe, calculate the base indicator
        for tf in self.timeframes:
            tf_relation = self._get_timeframe_relation(tf, self.reference_timeframe)

            if tf_relation == TimeframeRelation.SAME:
                # Check cache first
                cache_key = self._get_cache_key(data, tf)
                cached_result = self._get_from_cache(cache_key)

                if cached_result is not None:
                    tf_result = cached_result
                else:
                    # No conversion needed, calculate directly
                    tf_result = self._calculate_for_timeframe(result, tf)

                    # Cache the result
                    if self.enable_caching:
                        self._add_to_cache(cache_key, tf_result)

                # Get output columns from the base indicator
                indicator_columns = self._get_indicator_columns(tf_result)

                # Rename columns to include timeframe
                for col in indicator_columns:
                    result[f"{col}_{tf}"] = tf_result[col]

            elif tf_relation == TimeframeRelation.HIGHER:
                # Need to downsample the data
                downsampled_data = self._downsample_data(result, tf)

                # Check cache first
                cache_key = self._get_cache_key(downsampled_data, tf)
                cached_result = self._get_from_cache(cache_key)

                if cached_result is not None:
                    tf_result = cached_result
                else:
                    # Calculate indicator
                    tf_result = self._calculate_for_timeframe(downsampled_data, tf)

                    # Cache the result
                    if self.enable_caching:
                        self._add_to_cache(cache_key, tf_result)

                # Get output columns from the base indicator
                indicator_columns = self._get_indicator_columns(tf_result)

                # Merge results back to reference timeframe based on alignment method
                if self.alignment_method in ["upsample", "hybrid"]:
                    for col in indicator_columns:
                        # For upsampling, we forward fill the higher timeframe values
                        merged_result = self._merge_higher_timeframe(result.index, tf_result, col)
                        result[f"{col}_{tf}"] = merged_result

            elif tf_relation == TimeframeRelation.LOWER:
                # Need to upsample the data
                upsampled_data = self._upsample_data(result, tf)

                # Check cache first
                cache_key = self._get_cache_key(upsampled_data, tf)
                cached_result = self._get_from_cache(cache_key)

                if cached_result is not None:
                    tf_result = cached_result
                else:
                    # Calculate indicator
                    tf_result = self._calculate_for_timeframe(upsampled_data, tf)

                    # Cache the result
                    if self.enable_caching:
                        self._add_to_cache(cache_key, tf_result)

                # Get output columns from the base indicator
                indicator_columns = self._get_indicator_columns(tf_result)

                # Merge results back to reference timeframe based on alignment method
                if self.alignment_method in ["downsample", "hybrid"]:
                    for col in indicator_columns:
                        # For downsampling, we aggregate the lower timeframe values
                        merged_result = self._merge_lower_timeframe(result.index, tf_result, col)
                        result[f"{col}_{tf}"] = merged_result
                elif self.alignment_method == "upsample":
                    # Just take the most recent value for each reference bar
                    for col in indicator_columns:
                        # Resample back to reference timeframe
                        resampled = tf_result[col].resample(self._parse_timeframe_to_pandas(
                            self.reference_timeframe)).last()
                        # Align with original index
                        result[f"{col}_{tf}"] = resampled.reindex(result.index, method='ffill')

        return result

    def _calculate_for_timeframe(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Calculate the base indicator for the specified timeframe."""
        # Clone the index to preserve it
        index = data.index

        # Calculate the indicator
        result = self.base_indicator.calculate(data.copy())

        # Ensure the index is preserved
        if hasattr(result, 'index') and len(result.index) == len(index):
            result.index = index

        return result

    def _get_indicator_columns(self, data: pd.DataFrame) -> List[str]:
        """Extract the columns added by the base indicator."""
        # This is a heuristic - we assume the base indicator adds columns
        # that don't match the standard OHLCV column names
        standard_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']

        # Get columns that don't match standard OHLCV names
        indicator_cols = [col for col in data.columns if col.lower() not in standard_cols]

        return indicator_cols

    def _get_lowest_timeframe(self, timeframes: List[str]) -> str:
        """Find the lowest (shortest period) timeframe from the list."""
        # Convert timeframes to minutes for easy comparison
        tf_minutes = [self._timeframe_to_minutes(tf) for tf in timeframes]

        # Return the timeframe with the minimum minutes
        min_idx = tf_minutes.index(min(tf_minutes))
        return timeframes[min_idx]

    def _sort_timeframes_by_size(self, timeframes: List[str], ascending: bool = False) -> List[str]:
        """
        Sort timeframes by size (duration).

        Args:
            timeframes: List of timeframes to sort
            ascending: If True, sort from smallest to largest

        Returns:
            Sorted list of timeframes
        """
        # Convert timeframes to minutes for comparison
        tf_with_minutes = [(tf, self._timeframe_to_minutes(tf)) for tf in timeframes]

        # Sort by minutes
        sorted_tfs = sorted(tf_with_minutes, key=lambda x: x[1], reverse=not ascending)

        # Return just the timeframes
        return [tf for tf, _ in sorted_tfs]

    def _find_parent_timeframe(self, timeframe: str, available_timeframes: List[str]) -> Optional[str]:
        """
        Find a parent timeframe that can be used to derive the given timeframe.

        Args:
            timeframe: Timeframe to find a parent for
            available_timeframes: List of available timeframes

        Returns:
            Parent timeframe if found, None otherwise
        """
        if not available_timeframes:
            return None

        # Get the minutes for the target timeframe
        tf_minutes = self._timeframe_to_minutes(timeframe)

        # Find potential parent timeframes (larger timeframes that are multiples of this one)
        potential_parents = []

        for parent_tf in available_timeframes:
            parent_minutes = self._timeframe_to_minutes(parent_tf)

            # Check if parent is larger and is a multiple of the target
            if parent_minutes > tf_minutes and parent_minutes % tf_minutes == 0:
                potential_parents.append((parent_tf, parent_minutes))

        # If we found potential parents, return the smallest one
        if potential_parents:
            # Sort by minutes (ascending)
            sorted_parents = sorted(potential_parents, key=lambda x: x[1])
            return sorted_parents[0][0]

        return None

    def _derive_from_parent(
        self,
        data: pd.DataFrame,
        timeframe: str,
        parent_timeframe: str,
        parent_result: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Derive indicator values for a timeframe from a parent timeframe.

        Args:
            data: Original data at reference timeframe
            timeframe: Target timeframe
            parent_timeframe: Parent timeframe
            parent_result: Parent timeframe indicator results

        Returns:
            Derived indicator values for the target timeframe
        """
        # Check if the indicator supports hierarchical computation
        if hasattr(self.base_indicator, 'derive_from_parent'):
            # Use the indicator's own derivation method
            return self.base_indicator.derive_from_parent(
                data, timeframe, parent_timeframe, parent_result
            )

        # Default implementation: downsample the parent result to the target timeframe
        # This is a simplified approach - actual derivation depends on the indicator

        # Get the pandas frequency string for the target timeframe
        target_freq = self._parse_timeframe_to_pandas(timeframe)

        # Get indicator columns from parent result
        indicator_columns = self._get_indicator_columns(parent_result)

        # Create a copy of the parent result
        result = parent_result.copy()

        # Resample to the target timeframe
        # This is a simple approach - more sophisticated methods could be used
        resampled = result[indicator_columns].resample(target_freq).interpolate(method='linear')

        return resampled

    def _merge_results(self, result: pd.DataFrame, timeframe: str, tf_result: pd.DataFrame) -> None:
        """
        Merge timeframe results into the output DataFrame.

        Args:
            result: Output DataFrame to merge results into
            timeframe: Timeframe of the results
            tf_result: Indicator results for the timeframe
        """
        # Get output columns from the base indicator
        indicator_columns = self._get_indicator_columns(tf_result)

        # Get the relation between this timeframe and the reference
        tf_relation = self._get_timeframe_relation(timeframe, self.reference_timeframe)

        if tf_relation == TimeframeRelation.SAME:
            # Direct copy for same timeframe
            for col in indicator_columns:
                result[f"{col}_{timeframe}"] = tf_result[col]
        elif tf_relation == TimeframeRelation.HIGHER:
            # Merge higher timeframe results
            if self.alignment_method in ["upsample", "hybrid"]:
                for col in indicator_columns:
                    # For upsampling, we forward fill the higher timeframe values
                    merged_result = self._merge_higher_timeframe(result.index, tf_result, col)
                    result[f"{col}_{timeframe}"] = merged_result
        elif tf_relation == TimeframeRelation.LOWER:
            # Merge lower timeframe results
            if self.alignment_method in ["downsample", "hybrid"]:
                for col in indicator_columns:
                    # For downsampling, we aggregate the lower timeframe values
                    merged_result = self._merge_lower_timeframe(result.index, tf_result, col)
                    result[f"{col}_{timeframe}"] = merged_result
            elif self.alignment_method == "upsample":
                # Just take the most recent value for each reference bar
                for col in indicator_columns:
                    # Resample back to reference timeframe
                    resampled = tf_result[col].resample(self._parse_timeframe_to_pandas(
                        self.reference_timeframe)).last()
                    # Align with original index
                    result[f"{col}_{timeframe}"] = resampled.reindex(result.index, method='ffill')

    def _get_cache_key(self, data: pd.DataFrame, timeframe: str) -> str:
        """
        Generate a cache key for the given data and timeframe.

        Args:
            data: Input data
            timeframe: Timeframe

        Returns:
            Cache key string
        """
        # Use the indicator name, timeframe, and data shape/range as the key
        indicator_name = self.base_indicator.__class__.__name__

        # Include data range in the key
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

    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Get a result from the cache if available and not expired.

        Args:
            cache_key: Cache key

        Returns:
            Cached result if available, None otherwise
        """
        if not self.enable_caching:
            return None

        if cache_key in self._result_cache:
            # Check if the cache entry has expired
            timestamp = self._cache_timestamps.get(cache_key)
            if timestamp:
                current_time = pd.Timestamp.now()
                if (current_time - timestamp).total_seconds() < self.cache_ttl:
                    # Cache is still valid
                    return self._result_cache[cache_key]

        return None

    def _add_to_cache(self, cache_key: str, result: pd.DataFrame) -> None:
        """
        Add a result to the cache.

        Args:
            cache_key: Cache key
            result: Result to cache
        """
        if not self.enable_caching:
            return

        # Add to cache
        self._result_cache[cache_key] = result
        self._cache_timestamps[cache_key] = pd.Timestamp.now()

        # Clean up cache if it's too large
        if len(self._result_cache) > self._cache_max_size:
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
            if key in self._result_cache:
                del self._result_cache[key]
            if key in self._cache_timestamps:
                del self._cache_timestamps[key]

    def _timeframe_to_minutes(self, timeframe: str) -> float:
        """Convert a timeframe string to minutes."""
        if not timeframe:
            return 0

        # Remove any non-alphanumeric characters
        tf = ''.join(char for char in timeframe if char.isalnum())

        # Extract numeric value and unit
        if tf[-1].isalpha():
            value = float(tf[:-1]) if tf[:-1] else 1
            unit = tf[-1].lower()
        else:
            value = float(tf)
            unit = 'm'  # Default to minutes

        # Convert to minutes
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 60 * 24
        elif unit == 'w':
            return value * 60 * 24 * 7
        elif unit == 'M':
            return value * 60 * 24 * 30  # Approximate
        else:
            return value  # Default to original value

    def _parse_timeframe_to_pandas(self, timeframe: str) -> str:
        """Convert a timeframe string to pandas frequency string."""
        # Remove any non-alphanumeric characters
        tf = ''.join(char for char in timeframe if char.isalnum())

        # Extract numeric value and unit
        if tf[-1].isalpha():
            value = tf[:-1] if tf[:-1] else '1'
            unit = tf[-1].lower()
        else:
            value = tf
            unit = 'm'  # Default to minutes

        # Convert to pandas frequency string
        if unit == 'm':
            return f"{value}min"
        elif unit == 'h':
            return f"{value}H"
        elif unit == 'd':
            return f"{value}D"
        elif unit == 'w':
            return f"{value}W"
        elif unit == 'M':
            return f"{value}M"
        else:
            return f"{value}min"  # Default to minutes

    def _get_timeframe_relation(self, timeframe: str, reference: str) -> TimeframeRelation:
        """Determine the relation between a timeframe and the reference timeframe."""
        tf_minutes = self._timeframe_to_minutes(timeframe)
        ref_minutes = self._timeframe_to_minutes(reference)

        if tf_minutes < ref_minutes:
            return TimeframeRelation.LOWER
        elif tf_minutes > ref_minutes:
            return TimeframeRelation.HIGHER
        else:
            return TimeframeRelation.SAME

    def _downsample_data(self, data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Downsample data to a higher timeframe."""
        # Make a copy to avoid modifying the original data
        df = data.copy()

        # Get the pandas frequency string for the target timeframe
        freq = self._parse_timeframe_to_pandas(target_timeframe)

        # Create OHLCV aggregation
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }

        # Add volume aggregation if it exists
        if 'volume' in df.columns:
            agg_dict['volume'] = 'sum'

        # Resample and aggregate
        resampled = df.resample(freq).agg(agg_dict)

        # Drop any NaN values (can happen at the edges of the resampling)
        return resampled.dropna()

    def _upsample_data(self, data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Upsample data to a lower timeframe."""
        # This is a simplified approach - upsampling OHLCV data properly would
        # require fetching actual lower timeframe data
        warnings.warn(
            "Upsampling OHLCV data is not recommended. "
            "Results may not be accurate without actual lower timeframe data."
        )

        # Make a copy to avoid modifying the original data
        df = data.copy()

        # Get the pandas frequency string for the target timeframe
        freq = self._parse_timeframe_to_pandas(target_timeframe)

        # Create a new index with the target timeframe frequency
        start_time = df.index[0]
        end_time = df.index[-1]
        new_index = pd.date_range(start=start_time, end=end_time, freq=freq)

        # Reindex and forward fill values
        # This is not ideal for OHLCV data but is a simple approximation
        upsampled = df.reindex(new_index, method='ffill')

        # Drop any NaN values (can happen at the edges of the resampling)
        return upsampled.dropna()

    def _merge_higher_timeframe(
        self, target_index: pd.DatetimeIndex, source_data: pd.DataFrame, column_name: str
    ) -> pd.Series:
        """Merge higher timeframe indicator values to the target index."""
        # Create a Series with the source data
        source_series = source_data[column_name]

        # Create a new series with the target index
        result = pd.Series(index=target_index, dtype=source_series.dtype)

        # For each bar in the source (higher timeframe)
        for i in range(len(source_series)):
            if i < len(source_series) - 1:
                # Get the time range for this bar
                start_time = source_series.index[i]
                end_time = source_series.index[i + 1]

                # Find all target bars in this range
                mask = (target_index >= start_time) & (target_index < end_time)

                # Set the value for these bars
                result[mask] = source_series.iloc[i]
            else:
                # For the last bar, just fill forward
                mask = target_index >= source_series.index[i]
                result[mask] = source_series.iloc[i]

        return result

    def _merge_lower_timeframe(
        self, target_index: pd.DatetimeIndex, source_data: pd.DataFrame, column_name: str
    ) -> pd.Series:
        """Merge lower timeframe indicator values to the target index."""
        # Create a Series with the source data
        source_series = source_data[column_name]

        # Create a new series with the target index
        result = pd.Series(index=target_index, dtype=source_series.dtype)

        # For each bar in the target (higher) timeframe
        for i in range(len(target_index)):
            if i < len(target_index) - 1:
                # Get the time range for this bar
                start_time = target_index[i]
                end_time = target_index[i + 1]

                # Find all source bars in this range
                mask = (source_series.index >= start_time) & (source_series.index < end_time)
                values = source_series[mask]

                # If we have values, calculate the mean (or other aggregation)
                if len(values) > 0:
                    result[target_index[i]] = values.mean()
            else:
                # For the last bar, just take the mean of all remaining source bars
                mask = source_series.index >= target_index[i]
                values = source_series[mask]

                # If we have values, calculate the mean
                if len(values) > 0:
                    result[target_index[i]] = values.mean()

        return result

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Multi-Timeframe Indicator',
            'description': 'Calculates an indicator across multiple timeframes',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'base_indicator',
                    'description': 'Base indicator instance to apply across timeframes',
                    'type': 'object',
                    'default': None
                },
                {
                    'name': 'timeframes',
                    'description': 'List of timeframes to analyze',
                    'type': 'list',
                    'default': []
                },
                {
                    'name': 'reference_timeframe',
                    'description': 'Timeframe to align results to',
                    'type': 'string',
                    'default': None
                },
                {
                    'name': 'alignment_method',
                    'description': 'Method for aligning timeframes',
                    'type': 'string',
                    'default': 'upsample',
                    'options': ['upsample', 'downsample', 'hybrid']
                }
            ]
        }


class TimeframeComparison(BaseIndicator):
    """
    Timeframe Comparison Indicator

    Compares the same indicator across multiple timeframes to generate
    signals based on cross-timeframe confirmation or divergence.
    """

    category = "multi_timeframe"

    def __init__(
        self,
        indicator_name: str,
        timeframes: List[str],
        comparison_type: str = "trend_alignment",
        **kwargs
    ):
        """
        Initialize Timeframe Comparison Indicator.

        Args:
            indicator_name: Base name of the indicator to compare (without timeframe suffix)
            timeframes: List of timeframes to compare
            comparison_type: Type of comparison to make ("trend_alignment", "divergence", "momentum")
            **kwargs: Additional parameters
        """
        self.indicator_name = indicator_name
        self.timeframes = timeframes
        self.comparison_type = comparison_type

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate timeframe comparison for the given data.

        Args:
            data: DataFrame with indicator values across multiple timeframes

        Returns:
            DataFrame with comparison results
        """
        # Check if all required indicator columns are present
        required_columns = [f"{self.indicator_name}_{tf}" for tf in self.timeframes]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(
                f"Missing indicator columns: {missing_columns}. "
                "Make sure to calculate the indicator for all specified timeframes first."
            )

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Perform the comparison based on the selected type
        if self.comparison_type == "trend_alignment":
            result = self._calculate_trend_alignment(result)
        elif self.comparison_type == "divergence":
            result = self._calculate_divergence(result)
        elif self.comparison_type == "momentum":
            result = self._calculate_momentum(result)

        return result

    def _calculate_trend_alignment(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend alignment across timeframes."""
        df = data.copy()

        # For each timeframe, determine the trend direction
        trend_directions = {}

        for tf in self.timeframes:
            col = f"{self.indicator_name}_{tf}"

            # Simple trend direction: positive vs negative indicator values
            # This is a simplification - actual logic depends on the indicator
            trend_directions[tf] = np.sign(df[col])

        # Calculate alignment score (-1 to 1, where 1 is perfect alignment)
        alignment_score = np.zeros(len(df))

        for i in range(len(df)):
            # Get trend directions for this bar
            trends = [trend_directions[tf][i] for tf in self.timeframes]

            # Filter out NaN values
            valid_trends = [t for t in trends if not np.isnan(t)]

            if valid_trends:
                # Calculate alignment (average direction, -1 to 1)
                alignment_score[i] = sum(valid_trends) / len(valid_trends)

        # Add alignment score to result
        df[f"{self.indicator_name}_tf_alignment"] = alignment_score

        # Add binary signals for strong alignment
        df[f"{self.indicator_name}_bullish_alignment"] = (alignment_score > 0.5).astype(int)
        df[f"{self.indicator_name}_bearish_alignment"] = (alignment_score < -0.5).astype(int)

        return df

    def _calculate_divergence(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate divergence across timeframes."""
        df = data.copy()

        # Calculate the correlation between the lowest and highest timeframes
        lowest_tf = self.timeframes[0]
        highest_tf = self.timeframes[-1]

        lowest_col = f"{self.indicator_name}_{lowest_tf}"
        highest_col = f"{self.indicator_name}_{highest_tf}"

        # Calculate rolling correlation with a 10-bar window (adjustable)
        window = 10

        if len(df) >= window:
            # Calculate correlation
            correlation = df[lowest_col].rolling(window).corr(df[highest_col])

            # Calculate divergence score (1 - correlation, so 0 = perfect correlation, 2 = perfect negative correlation)
            df[f"{self.indicator_name}_tf_divergence"] = 1 - correlation

            # Add binary signals for significant divergence
            df[f"{self.indicator_name}_significant_divergence"] = (df[f"{self.indicator_name}_tf_divergence"] > 0.5).astype(int)
        else:
            # Not enough data for correlation
            df[f"{self.indicator_name}_tf_divergence"] = np.nan
            df[f"{self.indicator_name}_significant_divergence"] = 0

        return df

    def _calculate_momentum(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum confluence across timeframes."""
        df = data.copy()

        # For each timeframe, calculate momentum (direction of change)
        momentum_signals = {}

        for tf in self.timeframes:
            col = f"{self.indicator_name}_{tf}"

            # Calculate momentum (1-bar change)
            momentum = df[col].diff()

            # Normalize to -1, 0, 1
            momentum_signals[tf] = np.sign(momentum)

        # Calculate momentum score (average momentum across timeframes)
        momentum_score = np.zeros(len(df))

        for i in range(len(df)):
            # Get momentum signals for this bar
            signals = [momentum_signals[tf][i] for tf in self.timeframes]

            # Filter out NaN values
            valid_signals = [s for s in signals if not np.isnan(s)]

            if valid_signals:
                # Calculate average momentum (-1 to 1)
                momentum_score[i] = sum(valid_signals) / len(valid_signals)

        # Add momentum score to result
        df[f"{self.indicator_name}_tf_momentum"] = momentum_score

        # Add binary signals for strong momentum
        df[f"{self.indicator_name}_bullish_momentum"] = (momentum_score > 0.5).astype(int)
        df[f"{self.indicator_name}_bearish_momentum"] = (momentum_score < -0.5).astype(int)

        return df

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Timeframe Comparison',
            'description': 'Compares indicator values across multiple timeframes',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'indicator_name',
                    'description': 'Base name of the indicator to compare',
                    'type': 'string',
                    'default': ""
                },
                {
                    'name': 'timeframes',
                    'description': 'List of timeframes to compare',
                    'type': 'list',
                    'default': []
                },
                {
                    'name': 'comparison_type',
                    'description': 'Type of comparison to make',
                    'type': 'string',
                    'default': 'trend_alignment',
                    'options': ['trend_alignment', 'divergence', 'momentum']
                }
            ]
        }


class TimeframeConfluenceScanner(BaseIndicator):
    """
    Timeframe Confluence Scanner

    Scans multiple indicators across multiple timeframes to find
    confluence of signals, which can indicate stronger trading opportunities.
    """

    category = "multi_timeframe"

    def __init__(
        self,
        indicator_configs: List[Dict[str, Any]],
        timeframes: List[str],
        confluence_threshold: float = 0.6,
        lookback_period: int = 3,
        **kwargs
    ):
        """
        Initialize Timeframe Confluence Scanner.

        Args:
            indicator_configs: List of dictionaries with indicator configurations
                Each dict should have 'name', 'signal_column', and 'direction' (+1 or -1)
            timeframes: List of timeframes to analyze
            confluence_threshold: Threshold for significant confluence (0.0 to 1.0)
            lookback_period: Bars to look back for signal persistence
            **kwargs: Additional parameters
        """
        self.indicator_configs = indicator_configs
        self.timeframes = timeframes
        self.confluence_threshold = min(1.0, max(0.0, confluence_threshold))
        self.lookback_period = max(1, lookback_period)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate timeframe confluence for the given data.

        Args:
            data: DataFrame with indicator values across multiple timeframes

        Returns:
            DataFrame with confluence results
        """
        # Verify the data contains the necessary indicator columns
        for config in self.indicator_configs:
            for tf in self.timeframes:
                col = f"{config['signal_column']}_{tf}"
                if col not in data.columns:
                    raise ValueError(
                        f"Missing indicator column: {col}. "
                        "Make sure all indicators are calculated for all timeframes."
                    )

        # Make a copy to avoid modifying the input data
        result = data.copy()

        # Initialize confluence columns
        result['bullish_confluence'] = 0.0
        result['bearish_confluence'] = 0.0
        result['bullish_signal'] = 0
        result['bearish_signal'] = 0

        # Calculate confluence scores
        for i in range(len(result)):
            bullish_signals = 0
            bearish_signals = 0
            total_signals = len(self.indicator_configs) * len(self.timeframes)

            # Count bullish and bearish signals across all indicators and timeframes
            for config in self.indicator_configs:
                for tf in self.timeframes:
                    col = f"{config['signal_column']}_{tf}"

                    # Get signal value
                    signal_value = result[col].iloc[i]

                    # Skip NaN values
                    if pd.isna(signal_value):
                        total_signals -= 1
                        continue

                    # Check if signal is bullish or bearish based on configuration
                    direction = config_manager.get('direction', 1)  # Default to 1 (higher is bullish)

                    if (direction > 0 and signal_value > 0) or (direction < 0 and signal_value < 0):
                        bullish_signals += 1
                    elif (direction > 0 and signal_value < 0) or (direction < 0 and signal_value > 0):
                        bearish_signals += 1

            # Calculate confluence scores (percentage of signals in each direction)
            if total_signals > 0:
                bullish_score = bullish_signals / total_signals
                bearish_score = bearish_signals / total_signals

                result['bullish_confluence'].iloc[i] = bullish_score
                result['bearish_confluence'].iloc[i] = bearish_score

                # Generate binary signals based on confluence threshold
                if bullish_score >= self.confluence_threshold:
                    # Check if the confluence is persistent (appears in previous bars)
                    if i >= self.lookback_period:
                        lookback_scores = result['bullish_confluence'].iloc[i-self.lookback_period:i]
                        if all(score >= self.confluence_threshold * 0.8 for score in lookback_scores):
                            result['bullish_signal'].iloc[i] = 1
                    else:
                        # Not enough lookback data
                        result['bullish_signal'].iloc[i] = 1

                if bearish_score >= self.confluence_threshold:
                    # Check if the confluence is persistent (appears in previous bars)
                    if i >= self.lookback_period:
                        lookback_scores = result['bearish_confluence'].iloc[i-self.lookback_period:i]
                        if all(score >= self.confluence_threshold * 0.8 for score in lookback_scores):
                            result['bearish_signal'].iloc[i] = 1
                    else:
                        # Not enough lookback data
                        result['bearish_signal'].iloc[i] = 1

        return result

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            'name': 'Timeframe Confluence Scanner',
            'description': 'Scans for confluence of signals across indicators and timeframes',
            'category': cls.category,
            'parameters': [
                {
                    'name': 'indicator_configs',
                    'description': 'List of indicator configurations to scan',
                    'type': 'list',
                    'default': []
                },
                {
                    'name': 'timeframes',
                    'description': 'List of timeframes to analyze',
                    'type': 'list',
                    'default': []
                },
                {
                    'name': 'confluence_threshold',
                    'description': 'Threshold for significant confluence',
                    'type': 'float',
                    'default': 0.6
                },
                {
                    'name': 'lookback_period',
                    'description': 'Bars to look back for signal persistence',
                    'type': 'int',
                    'default': 3
                }
            ]
        }
