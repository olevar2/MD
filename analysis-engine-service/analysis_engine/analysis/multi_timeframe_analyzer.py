"""
Multi-Timeframe Analysis Module

This module provides tools for analyzing forex price data across multiple timeframes,
identifying alignment, and confirming signals through timeframe confluence.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum

from analysis_engine.analysis.advanced_ta.base import (
    AdvancedAnalysisBase,
    ConfidenceLevel,
    MarketDirection,
    AnalysisTimeframe
)


class TimeframeAlignment(Enum):
    """Possible alignments across timeframes"""
    STRONGLY_BULLISH = "strongly_bullish"  # All timeframes show bullish signals
    WEAKLY_BULLISH = "weakly_bullish"  # Most timeframes show bullish signals
    MIXED = "mixed"  # No clear alignment
    WEAKLY_BEARISH = "weakly_bearish"  # Most timeframes show bearish signals
    STRONGLY_BEARISH = "strongly_bearish"  # All timeframes show bearish signals


class MultiTimeFrameAnalyzer(AdvancedAnalysisBase):
    """
    Multi-timeframe analysis for forex price data

    This class analyzes indicators and patterns across multiple timeframes
    to identify alignment and confirm trading signals.
    Optimized for performance with caching and parallel processing.
    """

    # Class-level cache for analysis results
    _analysis_cache = {}
    _cache_timestamps = {}
    _cache_max_size = 50

    def __init__(
        self,
        timeframes: List[str] = None,
        indicators: List[str] = None,
        enable_caching: bool = True,
        cache_ttl: int = 300,  # 5 minutes
        use_parallel_processing: bool = True
    ):
        """
        Initialize the multi-timeframe analyzer with optimized performance

        Args:
            timeframes: List of timeframes to analyze (e.g., ['1h', '4h', '1d'])
            indicators: List of indicators to use in the analysis
            enable_caching: If True, cache analysis results for improved performance
            cache_ttl: Time-to-live for cached results in seconds
            use_parallel_processing: If True, use parallel processing for better performance
        """
        # Default timeframes and indicators if not provided
        default_timeframes = ["5m", "15m", "1h", "4h", "1d"]
        default_indicators = ["ma", "rsi", "macd"]

        parameters = {
            "timeframes": timeframes or default_timeframes,
            "indicators": indicators or default_indicators,
            "weight_higher_timeframes": True,
            "alignment_threshold": 0.7,  # Threshold for determining alignment
            "enable_caching": enable_caching,
            "cache_ttl": cache_ttl,
            "use_parallel_processing": use_parallel_processing
        }
        super().__init__("Multi-Timeframe Analysis", parameters)

        # Pre-sort timeframes by size for efficiency
        self.sorted_timeframes = sorted(
            self.parameters["timeframes"],
            key=lambda x: self._timeframe_to_minutes(x)
        )

        # Initialize indicator analysis functions map
        self.indicator_analyzers = {
            "ma": self._analyze_moving_averages,
            "rsi": self._analyze_rsi,
            "macd": self._analyze_macd
        }

    def analyze(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze price data across multiple timeframes with optimized performance

        Args:
            data_dict: Dictionary mapping timeframe strings to DataFrames

        Returns:
            Analysis results
        """
        # Validate input
        if not data_dict:
            return {"error": "No data provided for multi-timeframe analysis"}

        # Make sure we have all required timeframes
        missing_timeframes = [tf for tf in self.parameters["timeframes"] if tf not in data_dict]
        if missing_timeframes:
            return {
                "error": f"Missing data for timeframes: {missing_timeframes}",
                "available": list(data_dict.keys())
            }

        # Check cache first if enabled
        if self.parameters["enable_caching"]:
            cache_key = self._get_cache_key(data_dict)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

        # Initialize results dictionary
        results = {
            "timeframes_analyzed": list(data_dict.keys()),
            "indicator_signals": {},
            "trend_alignment": {},
            "overall_alignment": None,
            "confidence_level": None
        }

        # Use parallel processing if enabled
        if self.parameters["use_parallel_processing"]:
            # Process indicators in parallel
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Create tasks for each indicator
                indicator_tasks = {
                    indicator: executor.submit(self._analyze_indicator, data_dict, indicator)
                    for indicator in self.parameters["indicators"]
                }

                # Collect results
                for indicator, task in indicator_tasks.items():
                    results["indicator_signals"][indicator] = task.result()

                # Process trend alignment in parallel with indicators
                trend_task = executor.submit(self._determine_trend_alignment, data_dict)
                results["trend_alignment"] = trend_task.result()
        else:
            # Process sequentially
            # Analyze each indicator across timeframes
            for indicator in self.parameters["indicators"]:
                indicator_results = self._analyze_indicator(data_dict, indicator)
                results["indicator_signals"][indicator] = indicator_results

            # Identify trend alignment across timeframes
            results["trend_alignment"] = self._determine_trend_alignment(data_dict)

        # Calculate overall alignment and confidence
        results["overall_alignment"], results["confidence_level"] = self._calculate_overall_alignment(results)

        # Add single-direction binary signals
        results["bullish_signal"] = results["overall_alignment"] in [
            TimeframeAlignment.STRONGLY_BULLISH,
            TimeframeAlignment.WEAKLY_BULLISH
        ]

        results["bearish_signal"] = results["overall_alignment"] in [
            TimeframeAlignment.STRONGLY_BEARISH,
            TimeframeAlignment.WEAKLY_BEARISH
        ]

        # Add confirmation strength
        results["confirmation_strength"] = self._calculate_confirmation_strength(results)

        # Cache the result if enabled
        if self.parameters["enable_caching"]:
            self._add_to_cache(cache_key, results)

        return results

    def _analyze_indicator(self, data_dict: Dict[str, pd.DataFrame], indicator: str) -> Dict[str, Any]:
        """
        Analyze a specific indicator across timeframes

        Args:
            data_dict: Dictionary mapping timeframe strings to DataFrames
            indicator: Indicator to analyze (e.g., 'ma', 'rsi')

        Returns:
            Dictionary with indicator analysis results
        """
        # Use the function map for cleaner code and easier extensibility
        if indicator in self.indicator_analyzers:
            return self.indicator_analyzers[indicator](data_dict)
        else:
            # Return empty results for unknown indicators
            return {}

    def _analyze_moving_averages(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze moving averages across timeframes

        Args:
            data_dict: Dictionary mapping timeframe strings to DataFrames

        Returns:
            Dictionary with MA analysis results
        """
        ma_results = {}

        # Common MA periods to check
        fast_ma_period = 10
        slow_ma_period = 50

        for timeframe, df in data_dict.items():
            # Skip if this timeframe is not in our analysis parameters
            if timeframe not in self.parameters["timeframes"]:
                continue

            # Check if MA columns already exist
            fast_ma_col = f"ma_{fast_ma_period}"
            slow_ma_col = f"ma_{slow_ma_period}"

            # Calculate MAs if needed
            if fast_ma_col not in df.columns:
                df[fast_ma_col] = df["close"].rolling(window=fast_ma_period).mean()

            if slow_ma_col not in df.columns:
                df[slow_ma_col] = df["close"].rolling(window=slow_ma_period).mean()

            # Get latest values
            latest_fast = df[fast_ma_col].iloc[-1]
            latest_slow = df[slow_ma_col].iloc[-1]
            latest_close = df["close"].iloc[-1]

            # Determine trend direction
            if latest_fast > latest_slow:
                trend = "bullish"
                # Check if price is above fast MA (stronger bullish)
                if latest_close > latest_fast:
                    strength = "strong"
                else:
                    strength = "moderate"
            else:
                trend = "bearish"
                # Check if price is below fast MA (stronger bearish)
                if latest_close < latest_fast:
                    strength = "strong"
                else:
                    strength = "moderate"

            # Calculate distance between MAs as percentage
            ma_distance_pct = abs(latest_fast - latest_slow) / latest_slow * 100

            # Determine if crossover happened recently
            crossovers = []
            for i in range(max(5, len(df) - 10), len(df) - 1):
                prev_diff = df[fast_ma_col].iloc[i-1] - df[slow_ma_col].iloc[i-1]
                curr_diff = df[fast_ma_col].iloc[i] - df[slow_ma_col].iloc[i]

                if (prev_diff <= 0 and curr_diff > 0):
                    crossovers.append({"type": "bullish", "index": i})
                elif (prev_diff >= 0 and curr_diff < 0):
                    crossovers.append({"type": "bearish", "index": i})

            ma_results[timeframe] = {
                "trend": trend,
                "strength": strength,
                "fast_value": latest_fast,
                "slow_value": latest_slow,
                "ma_distance_pct": ma_distance_pct,
                "recent_crossovers": crossovers
            }

        return ma_results

    def _analyze_rsi(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze RSI across timeframes

        Args:
            data_dict: Dictionary mapping timeframe strings to DataFrames

        Returns:
            Dictionary with RSI analysis results
        """
        rsi_results = {}
        rsi_period = 14  # Standard RSI period

        for timeframe, df in data_dict.items():
            # Skip if this timeframe is not in our analysis parameters
            if timeframe not in self.parameters["timeframes"]:
                continue

            # Check if RSI column already exists
            rsi_col = f"rsi_{rsi_period}"

            # Calculate RSI if needed
            if rsi_col not in df.columns:
                # Calculate price changes
                delta = df["close"].diff()

                # Separate gains and losses
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)

                # Calculate average gain and loss
                avg_gain = gain.rolling(window=rsi_period).mean()
                avg_loss = loss.rolling(window=rsi_period).mean()

                # Calculate RS and RSI
                rs = avg_gain / avg_loss
                df[rsi_col] = 100 - (100 / (1 + rs))

            # Get latest RSI value
            latest_rsi = df[rsi_col].iloc[-1]

            # Determine trend and conditions
            if latest_rsi > 70:
                trend = "bearish"  # Overbought
                strength = "strong" if latest_rsi > 80 else "moderate"
            elif latest_rsi < 30:
                trend = "bullish"  # Oversold
                strength = "strong" if latest_rsi < 20 else "moderate"
            elif latest_rsi > 50:
                trend = "bullish"  # Above midpoint
                strength = "weak"
            else:
                trend = "bearish"  # Below midpoint
                strength = "weak"

            # Check for RSI divergence
            price_direction = "up" if df["close"].iloc[-1] > df["close"].iloc[-5] else "down"
            rsi_direction = "up" if df[rsi_col].iloc[-1] > df[rsi_col].iloc[-5] else "down"

            divergence = None
            if price_direction != rsi_direction:
                divergence = "bullish" if price_direction == "down" and rsi_direction == "up" else "bearish"

            rsi_results[timeframe] = {
                "value": latest_rsi,
                "trend": trend,
                "strength": strength,
                "condition": "overbought" if latest_rsi > 70 else "oversold" if latest_rsi < 30 else "neutral",
                "divergence": divergence
            }

        return rsi_results

    def _analyze_macd(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze MACD across timeframes

        Args:
            data_dict: Dictionary mapping timeframe strings to DataFrames

        Returns:
            Dictionary with MACD analysis results
        """
        macd_results = {}

        # MACD parameters
        fast_period = 12
        slow_period = 26
        signal_period = 9

        for timeframe, df in data_dict.items():
            # Skip if this timeframe is not in our analysis parameters
            if timeframe not in self.parameters["timeframes"]:
                continue

            # Check if MACD columns already exist
            macd_col = "macd"
            signal_col = "macd_signal"
            hist_col = "macd_hist"

            # Calculate MACD if needed
            if macd_col not in df.columns:
                # Calculate EMAs
                ema_fast = df["close"].ewm(span=fast_period, adjust=False).mean()
                ema_slow = df["close"].ewm(span=slow_period, adjust=False).mean()

                # Calculate MACD line
                df[macd_col] = ema_fast - ema_slow

                # Calculate signal line
                df[signal_col] = df[macd_col].ewm(span=signal_period, adjust=False).mean()

                # Calculate histogram
                df[hist_col] = df[macd_col] - df[signal_col]

            # Get latest values
            latest_macd = df[macd_col].iloc[-1]
            latest_signal = df[signal_col].iloc[-1]
            latest_hist = df[hist_col].iloc[-1]

            # Determine trend based on MACD and signal line relationship
            if latest_macd > latest_signal:
                trend = "bullish"
            else:
                trend = "bearish"

            # Determine strength based on histogram and absolute MACD value
            if abs(latest_hist) > abs(latest_macd * 0.1):
                strength = "strong"
            else:
                strength = "moderate"

            # Check for zero line cross
            zero_cross = None
            if (df[macd_col].iloc[-2] < 0 and latest_macd >= 0):
                zero_cross = "bullish"
            elif (df[macd_col].iloc[-2] > 0 and latest_macd <= 0):
                zero_cross = "bearish"

            # Check for signal line cross
            signal_cross = None
            if (df[macd_col].iloc[-2] < df[signal_col].iloc[-2] and
                latest_macd > latest_signal):
                signal_cross = "bullish"
            elif (df[macd_col].iloc[-2] > df[signal_col].iloc[-2] and
                   latest_macd < latest_signal):
                signal_cross = "bearish"

            macd_results[timeframe] = {
                "macd_value": latest_macd,
                "signal_value": latest_signal,
                "histogram": latest_hist,
                "trend": trend,
                "strength": strength,
                "zero_cross": zero_cross,
                "signal_cross": signal_cross
            }

        return macd_results

    def _determine_trend_alignment(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        Determine trend alignment across timeframes

        Args:
            data_dict: Dictionary mapping timeframe strings to DataFrames

        Returns:
            Dictionary with trend direction for each timeframe
        """
        trend_alignment = {}

        for timeframe, df in data_dict.items():
            # Skip if this timeframe is not in our analysis parameters
            if timeframe not in self.parameters["timeframes"]:
                continue

            # Use simple method - compare latest close with MA50
            ma50_col = "ma_50"
            if ma50_col not in df.columns:
                df[ma50_col] = df["close"].rolling(window=50).mean()

            latest_close = df["close"].iloc[-1]
            latest_ma50 = df[ma50_col].iloc[-1]

            # Determine basic trend
            if latest_close > latest_ma50:
                trend_direction = "bullish"
            else:
                trend_direction = "bearish"

            # Check trend strength
            distance_pct = abs(latest_close - latest_ma50) / latest_ma50 * 100
            if distance_pct > 2.0:
                strength = "strong"
            elif distance_pct > 0.5:
                strength = "moderate"
            else:
                strength = "weak"

            trend_alignment[timeframe] = {
                "direction": trend_direction,
                "strength": strength,
                "distance_pct": distance_pct
            }

        return trend_alignment

    def _calculate_overall_alignment(self, results: Dict[str, Any]) -> Tuple[TimeframeAlignment, ConfidenceLevel]:
        """
        Calculate overall timeframe alignment and confidence

        Args:
            results: Analysis results dictionary

        Returns:
            Tuple of (alignment, confidence_level)
        """
        # Count bullish and bearish signals across timeframes
        bullish_count = 0
        bearish_count = 0
        total_count = 0

        # Assign weights to timeframes (higher weight to longer timeframes if enabled)
        timeframe_weights = {}
        timeframes_sorted = sorted(
            self.parameters["timeframes"],
            key=lambda x: self._timeframe_to_minutes(x)
        )

        if self.parameters["weight_higher_timeframes"]:
            # Assign higher weights to longer timeframes
            for i, tf in enumerate(timeframes_sorted):
                timeframe_weights[tf] = (i + 1) / len(timeframes_sorted)
        else:
            # Equal weights
            for tf in timeframes_sorted:
                timeframe_weights[tf] = 1.0

        # Normalize weights
        weight_sum = sum(timeframe_weights.values())
        for tf in timeframe_weights:
            timeframe_weights[tf] /= weight_sum

        # Calculate weighted score
        alignment_score = 0.0  # Range: -1.0 (strongly bearish) to 1.0 (strongly bullish)

        # Process trend alignment
        for timeframe, data in results["trend_alignment"].items():
            if timeframe in timeframe_weights:
                weight = timeframe_weights[timeframe]
                direction = data["direction"]
                strength = data["strength"]

                strength_multiplier = 1.0
                if strength == "strong":
                    strength_multiplier = 1.0
                elif strength == "moderate":
                    strength_multiplier = 0.7
                else:  # weak
                    strength_multiplier = 0.4

                if direction == "bullish":
                    alignment_score += weight * strength_multiplier
                else:  # bearish
                    alignment_score -= weight * strength_multiplier

        # Determine alignment category
        if alignment_score > 0.7:
            alignment = TimeframeAlignment.STRONGLY_BULLISH
        elif alignment_score > 0.3:
            alignment = TimeframeAlignment.WEAKLY_BULLISH
        elif alignment_score < -0.7:
            alignment = TimeframeAlignment.STRONGLY_BEARISH
        elif alignment_score < -0.3:
            alignment = TimeframeAlignment.WEAKLY_BEARISH
        else:
            alignment = TimeframeAlignment.MIXED

        # Calculate confidence level based on alignment score
        abs_score = abs(alignment_score)
        if abs_score > 0.8:
            confidence = ConfidenceLevel.VERY_HIGH
        elif abs_score > 0.6:
            confidence = ConfidenceLevel.HIGH
        elif abs_score > 0.4:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW

        return alignment, confidence

    def _calculate_confirmation_strength(self, results: Dict[str, Any]) -> float:
        """
        Calculate the confirmation strength as a percentage

        Args:
            results: Analysis results dictionary

        Returns:
            Confirmation strength from 0.0 to 1.0
        """
        # Start with overall alignment score
        if results["overall_alignment"] == TimeframeAlignment.STRONGLY_BULLISH:
            base_strength = 1.0
        elif results["overall_alignment"] == TimeframeAlignment.WEAKLY_BULLISH:
            base_strength = 0.7
        elif results["overall_alignment"] == TimeframeAlignment.STRONGLY_BEARISH:
            base_strength = 1.0
        elif results["overall_alignment"] == TimeframeAlignment.WEAKLY_BEARISH:
            base_strength = 0.7
        else:  # Mixed
            base_strength = 0.3

        # Adjust based on timeframe count and indicator agreement
        indicator_agreement = 0.0

        # Check indicator alignment
        for indicator, results_by_tf in results["indicator_signals"].items():
            bullish_count = 0
            bearish_count = 0
            total_count = 0

            for tf, indicator_data in results_by_tf.items():
                trend = indicator_data.get("trend", "")
                if trend == "bullish":
                    bullish_count += 1
                elif trend == "bearish":
                    bearish_count += 1
                total_count += 1

            if total_count > 0:
                max_alignment = max(bullish_count, bearish_count) / total_count
                indicator_agreement += max_alignment

        # Average indicator agreement
        if results["indicator_signals"]:
            indicator_agreement /= len(results["indicator_signals"])

            # Combine with base strength
            final_strength = (base_strength * 0.7) + (indicator_agreement * 0.3)
        else:
            final_strength = base_strength

        return min(1.0, final_strength)

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """
        Convert timeframe string to minutes

        Args:
            timeframe: Timeframe string like '1h', '4h', '1d'

        Returns:
            Minutes representation
        """
        if not timeframe:
            return 0

        # Handle numeric part and unit
        if timeframe[-1].isdigit():
            return int(timeframe)  # Assume already in minutes

        try:
            # Extract numeric part
            num = int("".join([c for c in timeframe if c.isdigit()]))

            # Apply unit multiplier
            unit = timeframe[-1].lower()
            if unit == 'm':
                return num
            elif unit == 'h':
                return num * 60
            elif unit == 'd':
                return num * 60 * 24
            elif unit == 'w':
                return num * 60 * 24 * 7
            else:
                return num
        except:
            return 0

    def _get_cache_key(self, data_dict: Dict[str, pd.DataFrame]) -> str:
        """
        Generate a cache key for the given data.

        Args:
            data_dict: Dictionary of DataFrames by timeframe

        Returns:
            Cache key string
        """
        # Create a key based on the data shape and range for each timeframe
        key_parts = []

        for tf in sorted(data_dict.keys()):
            df = data_dict[tf]
            if len(df) > 0:
                # Use the last few timestamps and values to create a unique key
                last_idx = min(5, len(df))
                last_timestamps = [ts.isoformat() for ts in df.index[-last_idx:]]
                last_values = df["close"].iloc[-last_idx:].round(4).tolist()

                # Combine into a timeframe-specific key part
                tf_key = f"{tf}_{len(df)}_{last_timestamps[-1]}_{last_values[-1]}"
                key_parts.append(tf_key)

        # Add indicator and parameter information
        indicators_key = "_".join(sorted(self.parameters["indicators"]))
        params_key = f"w{int(self.parameters['weight_higher_timeframes'])}_t{self.parameters['alignment_threshold']}"

        # Combine all parts
        return f"mta_{indicators_key}_{params_key}_{'_'.join(key_parts)}"

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get a result from the cache if available and not expired.

        Args:
            cache_key: Cache key

        Returns:
            Cached result if available, None otherwise
        """
        if not self.parameters["enable_caching"]:
            return None

        if cache_key in self._analysis_cache:
            # Check if the cache entry has expired
            timestamp = self._cache_timestamps.get(cache_key)
            if timestamp:
                current_time = pd.Timestamp.now()
                if (current_time - timestamp).total_seconds() < self.parameters["cache_ttl"]:
                    # Cache is still valid
                    return self._analysis_cache[cache_key]

        return None

    def _add_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """
        Add a result to the cache.

        Args:
            cache_key: Cache key
            result: Result to cache
        """
        if not self.parameters["enable_caching"]:
            return

        # Add to cache
        self._analysis_cache[cache_key] = result
        self._cache_timestamps[cache_key] = pd.Timestamp.now()

        # Clean up cache if it's too large
        if len(self._analysis_cache) > self._cache_max_size:
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
            if key in self._analysis_cache:
                del self._analysis_cache[key]
            if key in self._cache_timestamps:
                del self._cache_timestamps[key]

    def update_incremental(self, data_dict: Dict[str, pd.DataFrame], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update analysis incrementally with new data

        Args:
            data_dict: Dictionary mapping timeframe strings to DataFrames with new data
            previous_results: Results from previous analysis

        Returns:
            Updated analysis results
        """
        # Check if we can use the previous results as a starting point
        if (previous_results and
            "timeframes_analyzed" in previous_results and
            set(previous_results["timeframes_analyzed"]) == set(data_dict.keys())):

            # Check if only a small amount of new data has been added
            # In that case, we might be able to update incrementally
            can_update_incrementally = True
            for tf in data_dict.keys():
                if tf in previous_results.get("last_analyzed_indices", {}):
                    last_idx = previous_results["last_analyzed_indices"][tf]
                    current_len = len(data_dict[tf])

                    # If more than 5% new data, do a full reanalysis
                    if current_len - last_idx > 0.05 * current_len:
                        can_update_incrementally = False
                        break
                else:
                    can_update_incrementally = False
                    break

            if can_update_incrementally:
                # Update only the latest data points
                updated_results = self._update_analysis(data_dict, previous_results)

                # Store the current indices for future incremental updates
                updated_results["last_analyzed_indices"] = {
                    tf: len(df) for tf, df in data_dict.items()
                }

                return updated_results

        # For most cases, we need to re-analyze fully
        # since the alignment across timeframes can change significantly with new data
        results = self.analyze(data_dict)

        # Store the current indices for future incremental updates
        results["last_analyzed_indices"] = {
            tf: len(df) for tf, df in data_dict.items()
        }

        return results

    def _update_analysis(self, data_dict: Dict[str, pd.DataFrame], previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update analysis results with new data incrementally.

        Args:
            data_dict: Dictionary mapping timeframe strings to DataFrames with new data
            previous_results: Results from previous analysis

        Returns:
            Updated analysis results
        """
        # Start with a copy of the previous results
        updated_results = previous_results.copy()

        # Update indicator signals with new data
        for indicator in self.parameters["indicators"]:
            if indicator in self.indicator_analyzers:
                # Get updated indicator analysis
                updated_indicator = self.indicator_analyzers[indicator](data_dict)
                updated_results["indicator_signals"][indicator] = updated_indicator

        # Update trend alignment
        updated_results["trend_alignment"] = self._determine_trend_alignment(data_dict)

        # Recalculate overall alignment and confidence
        updated_results["overall_alignment"], updated_results["confidence_level"] = self._calculate_overall_alignment(updated_results)

        # Update binary signals
        updated_results["bullish_signal"] = updated_results["overall_alignment"] in [
            TimeframeAlignment.STRONGLY_BULLISH,
            TimeframeAlignment.WEAKLY_BULLISH
        ]

        updated_results["bearish_signal"] = updated_results["overall_alignment"] in [
            TimeframeAlignment.STRONGLY_BEARISH,
            TimeframeAlignment.WEAKLY_BEARISH
        ]

        # Update confirmation strength
        updated_results["confirmation_strength"] = self._calculate_confirmation_strength(updated_results)

        return updated_results
