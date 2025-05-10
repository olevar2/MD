"""
Related Pairs Confluence Detector Module

This module provides functionality for detecting confluence signals across related currency pairs,
which can strengthen trading signals and improve prediction accuracy.

Part of Phase 2 implementation to enhance currency correlation analysis.

Optimized for performance with:
- Vectorized operations for faster calculations
- Caching of intermediate results
- Parallel processing for independent calculations
- Early termination for performance improvement
- Memory optimization
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
from functools import lru_cache
import concurrent.futures
from threading import Lock
import time

from analysis_engine.multi_asset.correlation_tracking_service import CorrelationTrackingService
from analysis_engine.multi_asset.currency_strength_analyzer import CurrencyStrengthAnalyzer


class RelatedPairsConfluenceAnalyzer:
    """
    Detects confluence signals across related currency pairs to strengthen trading signals.

    This service:
    - Identifies confirmation signals from correlated pairs
    - Detects divergence between related pairs that may indicate potential reversals
    - Provides multi-pair confirmation for trading signals
    """

    def __init__(
        self,
        correlation_service: Optional[CorrelationTrackingService] = None,
        currency_strength_analyzer: Optional[CurrencyStrengthAnalyzer] = None,
        correlation_threshold: float = 0.7,
        lookback_periods: int = 20,
        cache_ttl_minutes: int = 60,
        max_workers: int = 4
    ):
        """
        Initialize the related pairs confluence detector.

        Args:
            correlation_service: Service for tracking correlations between pairs
            currency_strength_analyzer: Service for analyzing currency strength
            correlation_threshold: Minimum correlation to consider pairs related
            lookback_periods: Number of periods to look back for analysis
            cache_ttl_minutes: Time-to-live for cache entries in minutes
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.correlation_service = correlation_service
        self.currency_strength_analyzer = currency_strength_analyzer
        self.correlation_threshold = correlation_threshold
        self.lookback_periods = lookback_periods
        self.cache_ttl_minutes = cache_ttl_minutes
        self.max_workers = max_workers

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Enhanced caching system
        self.related_pairs_cache = {}
        self.signal_cache = {}
        self.momentum_cache = {}
        self.cache_lock = Lock()
        self.last_cache_cleanup = time.time()

        # Performance metrics
        self.performance_metrics = {
            "find_related_pairs": [],
            "detect_confluence": [],
            "analyze_divergence": []
        }
        self.metrics_lock = Lock()

        self.logger.info(
            f"RelatedPairsConfluenceAnalyzer initialized with correlation threshold {correlation_threshold}, "
            f"cache TTL {cache_ttl_minutes} minutes, max workers {max_workers}"
        )

    def _clean_cache(self, force: bool = False) -> None:
        """Clean expired entries from all caches"""
        current_time = time.time()

        # Only clean periodically to avoid overhead
        if not force and current_time - self.last_cache_cleanup < 300:  # Clean at most once per 5 minutes
            return

        with self.cache_lock:
            # Calculate expiry time
            expiry_time = current_time - (self.cache_ttl_minutes * 60)

            # Clean related pairs cache
            expired_keys = []
            for key, (timestamp, _) in self.related_pairs_cache.items():
                if timestamp < expiry_time:
                    expired_keys.append(key)

            for key in expired_keys:
                del self.related_pairs_cache[key]

            # Clean signal cache
            expired_keys = []
            for key, (timestamp, _) in self.signal_cache.items():
                if timestamp < expiry_time:
                    expired_keys.append(key)

            for key in expired_keys:
                del self.signal_cache[key]

            # Clean momentum cache
            expired_keys = []
            for key, (timestamp, _) in self.momentum_cache.items():
                if timestamp < expiry_time:
                    expired_keys.append(key)

            for key in expired_keys:
                del self.momentum_cache[key]

            # Update last cleanup time
            self.last_cache_cleanup = current_time

            # Log cache stats
            self.logger.debug(
                f"Cache cleaned: {len(self.related_pairs_cache)} related pairs, "
                f"{len(self.signal_cache)} signals, {len(self.momentum_cache)} momentum entries"
            )

    async def find_related_pairs(self, symbol: str, min_correlation: Optional[float] = None) -> Dict[str, float]:
        """
        Find pairs that are correlated with the given symbol.

        Optimized with:
        - Enhanced caching with TTL
        - Performance monitoring
        - Early termination for invalid inputs

        Args:
            symbol: The currency pair to find related pairs for
            min_correlation: Optional minimum correlation threshold (overrides default)

        Returns:
            Dictionary mapping related pairs to their correlation values
        """
        start_time = time.time()

        # Early termination for invalid inputs
        if not symbol or not self.correlation_service:
            self.logger.warning("No correlation service provided or invalid symbol, cannot find related pairs")
            return {}

        # Use provided threshold or default
        threshold = min_correlation if min_correlation is not None else self.correlation_threshold

        # Clean cache periodically
        self._clean_cache()

        # Check cache with thread safety
        cache_key = f"{symbol}_{threshold}"
        with self.cache_lock:
            if cache_key in self.related_pairs_cache:
                timestamp, cached_result = self.related_pairs_cache[cache_key]
                if time.time() - timestamp < (self.cache_ttl_minutes * 60):
                    # Record performance metric
                    execution_time = time.time() - start_time
                    with self.metrics_lock:
                        self.performance_metrics["find_related_pairs"].append(execution_time)

                    self.logger.debug(f"Cache hit for {cache_key}, returned in {execution_time:.4f}s")
                    return cached_result.copy()  # Return a copy to prevent modification of cached data

        # Get correlation matrix from service
        try:
            # This assumes the correlation service has a method to get all correlations
            t0 = time.time()
            all_correlations = await self.correlation_service.get_all_correlations()
            correlation_fetch_time = time.time() - t0

            # Early termination if no correlations
            if not all_correlations:
                self.logger.warning("No correlations returned from service")
                return {}

            # Filter for correlations with our symbol using vectorized operations if possible
            t0 = time.time()
            related = {}

            # Try to use vectorized operations if the data structure allows
            if isinstance(all_correlations, pd.DataFrame):
                # If it's a DataFrame, we can use vectorized operations
                if symbol in all_correlations.columns:
                    mask = abs(all_correlations[symbol]) >= threshold
                    related = all_correlations.loc[mask, symbol].to_dict()
            else:
                # Otherwise, use the traditional approach
                for pair, correlations in all_correlations.items():
                    if symbol in correlations:
                        corr_value = correlations[symbol]
                        if abs(corr_value) >= threshold:
                            related[pair] = corr_value

            filtering_time = time.time() - t0

            # Update cache with thread safety
            with self.cache_lock:
                self.related_pairs_cache[cache_key] = (time.time(), related.copy())

            # Record performance metric
            execution_time = time.time() - start_time
            with self.metrics_lock:
                self.performance_metrics["find_related_pairs"].append(execution_time)

            # Log performance details if it took longer than expected
            if execution_time > 0.1:  # Log if > 100ms
                self.logger.info(
                    f"find_related_pairs performance: total={execution_time:.4f}s, "
                    f"correlation_fetch={correlation_fetch_time:.4f}s, filtering={filtering_time:.4f}s, "
                    f"found {len(related)} related pairs for {symbol}"
                )

            return related

        except Exception as e:
            self.logger.error(f"Error finding related pairs: {e}", exc_info=True)
            return {}

    def detect_confluence(
        self,
        symbol: str,
        price_data: Dict[str, pd.DataFrame],
        signal_type: str,
        signal_direction: str,
        related_pairs: Optional[Dict[str, float]] = None,
        use_currency_strength: bool = True,
        min_confirmation_strength: float = 0.3
    ) -> Dict[str, Any]:
        """
        Detect confluence signals across related currency pairs.

        Optimized with:
        - Parallel processing for analyzing multiple pairs
        - Enhanced caching for signal detection
        - Early termination for invalid inputs
        - Performance monitoring
        - Vectorized operations where possible

        Args:
            symbol: The primary currency pair
            price_data: Dictionary mapping currency pairs to price DataFrames
            signal_type: Type of signal to look for (e.g., "trend", "reversal", "breakout")
            signal_direction: Direction of the signal ("bullish" or "bearish")
            related_pairs: Optional dictionary of related pairs (if not provided, will be calculated)
            use_currency_strength: Whether to incorporate currency strength analysis
            min_confirmation_strength: Minimum signal strength to consider as confirmation

        Returns:
            Dictionary with confluence analysis results
        """
        start_time = time.time()

        # Early termination for invalid inputs
        if symbol not in price_data or price_data[symbol].empty:
            return {"error": f"No price data available for {symbol}"}

        # Get related pairs if not provided
        if related_pairs is None:
            # Since this is a synchronous method, we can't use the async find_related_pairs
            # In a real implementation, you would either make this method async or use a synchronous version
            related_pairs = {}

        # Filter price data to include only related pairs that we have data for
        available_related = {
            pair: corr for pair, corr in related_pairs.items()
            if pair in price_data and not price_data[pair].empty
        }

        # Early termination if no related pairs
        if not available_related:
            return {
                "symbol": symbol,
                "signal_type": signal_type,
                "signal_direction": signal_direction,
                "related_pairs_count": 0,
                "confluence_score": 0.0,
                "message": "No related pairs with available data"
            }

        # Get currency strength data if available and requested
        currency_strength_data = {}
        divergence_signals = {}
        if use_currency_strength and self.currency_strength_analyzer:
            try:
                # Calculate currency strength
                t0 = time.time()
                currency_strength_data = self.currency_strength_analyzer.calculate_currency_strength(price_data)

                # Get divergence signals if available
                divergence_signals = self.currency_strength_analyzer.compute_divergence_signals(price_data)

                currency_strength_time = time.time() - t0
                if currency_strength_time > 0.1:  # Log if > 100ms
                    self.logger.debug(f"Currency strength calculation took {currency_strength_time:.4f}s")
            except Exception as e:
                self.logger.warning(f"Error calculating currency strength: {e}")

        # Use parallel processing to analyze related pairs
        confirmations = []
        contradictions = []
        neutral = []

        # Define a function to analyze a single pair
        def analyze_pair(related_pair, correlation):
            # Determine expected direction based on correlation
            expected_direction = signal_direction
            if correlation < 0:
                # For negative correlation, expect opposite direction
                expected_direction = "bullish" if signal_direction == "bearish" else "bearish"

            # Detect signal in related pair
            related_signal = self._detect_signal(
                price_data[related_pair],
                signal_type,
                self.lookback_periods
            )

            if not related_signal:
                return "neutral", {
                    "pair": related_pair,
                    "correlation": correlation,
                    "reason": "no_signal_detected"
                }

            # Check if direction matches expected and signal is strong enough
            if related_signal["direction"] == expected_direction and related_signal.get("strength", 0.0) >= min_confirmation_strength:
                return "confirmation", {
                    "pair": related_pair,
                    "correlation": correlation,
                    "signal_strength": related_signal.get("strength", 0.0),
                    "expected_direction": expected_direction,
                    "signal_type": related_signal.get("type", "unknown")
                }
            elif related_signal.get("strength", 0.0) >= min_confirmation_strength:
                return "contradiction", {
                    "pair": related_pair,
                    "correlation": correlation,
                    "signal_strength": related_signal.get("strength", 0.0),
                    "expected_direction": expected_direction,
                    "actual_direction": related_signal["direction"],
                    "signal_type": related_signal.get("type", "unknown")
                }
            else:
                # Signal too weak to be considered
                return "neutral", {
                    "pair": related_pair,
                    "correlation": correlation,
                    "signal_direction": related_signal["direction"],
                    "signal_strength": related_signal.get("strength", 0.0),
                    "reason": "weak_signal"
                }

        # Use parallel processing if we have enough pairs
        if len(available_related) >= 3 and self.max_workers > 1:
            t0 = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(available_related))) as executor:
                # Submit tasks for each related pair
                future_to_pair = {
                    executor.submit(analyze_pair, related_pair, correlation): (related_pair, correlation)
                    for related_pair, correlation in available_related.items()
                }

                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_pair):
                    try:
                        result_type, result_data = future.result()
                        if result_type == "confirmation":
                            confirmations.append(result_data)
                        elif result_type == "contradiction":
                            contradictions.append(result_data)
                        else:
                            neutral.append(result_data)
                    except Exception as e:
                        self.logger.error(f"Error analyzing pair: {e}")
                        pair_info = future_to_pair[future]
                        neutral.append({
                            "pair": pair_info[0],
                            "correlation": pair_info[1],
                            "reason": f"error: {str(e)}"
                        })

            parallel_time = time.time() - t0
            if parallel_time > 0.1:  # Log if > 100ms
                self.logger.debug(f"Parallel pair analysis took {parallel_time:.4f}s for {len(available_related)} pairs")
        else:
            # Sequential processing for small number of pairs
            for related_pair, correlation in available_related.items():
                try:
                    result_type, result_data = analyze_pair(related_pair, correlation)
                    if result_type == "confirmation":
                        confirmations.append(result_data)
                    elif result_type == "contradiction":
                        contradictions.append(result_data)
                    else:
                        neutral.append(result_data)
                except Exception as e:
                    self.logger.error(f"Error analyzing pair {related_pair}: {e}")
                    neutral.append({
                        "pair": related_pair,
                        "correlation": correlation,
                        "reason": f"error: {str(e)}"
                    })

        # Calculate confluence score
        t0 = time.time()
        total_pairs = len(available_related)
        confirmation_count = len(confirmations)
        contradiction_count = len(contradictions)
        neutral_count = len(neutral)

        # Use numpy for faster calculations if possible
        if confirmations:
            # Extract correlation and signal strength as numpy arrays
            conf_correlations = np.array([abs(conf["correlation"]) for conf in confirmations])
            conf_strengths = np.array([conf["signal_strength"] for conf in confirmations])

            # Calculate weighted confirmations using vectorized operations
            weighted_confirmations = np.sum(conf_correlations * conf_strengths)
        else:
            weighted_confirmations = 0.0

        if contradictions:
            # Extract correlation and signal strength as numpy arrays
            cont_correlations = np.array([abs(cont["correlation"]) for cont in contradictions])
            cont_strengths = np.array([cont["signal_strength"] for cont in contradictions])

            # Calculate weighted contradictions using vectorized operations
            weighted_contradictions = np.sum(cont_correlations * cont_strengths)
        else:
            weighted_contradictions = 0.0

        # Calculate final score (0.0 to 1.0)
        if total_pairs > 0:
            raw_score = (weighted_confirmations - weighted_contradictions) / total_pairs
            confluence_score = max(0.0, min(1.0, (raw_score + 1) / 2))  # Normalize to 0-1

            # Apply bonus for high confirmation percentage
            if confirmation_count > 0:
                confirmation_ratio = confirmation_count / (confirmation_count + contradiction_count) if (confirmation_count + contradiction_count) > 0 else 0
                if confirmation_ratio > 0.8:  # 80% or more confirmations
                    confluence_score = min(1.0, confluence_score * 1.2)  # 20% bonus
        else:
            confluence_score = 0.0

        score_calculation_time = time.time() - t0
        if score_calculation_time > 0.05:  # Log if > 50ms
            self.logger.debug(f"Score calculation took {score_calculation_time:.4f}s")

        # Incorporate currency strength if available
        currency_strength_confirms = None
        if currency_strength_data and len(currency_strength_data) > 0:
            try:
                # Extract currencies from the symbol
                pair_currencies = self._extract_currencies_from_pairs([symbol])
                if symbol in pair_currencies:
                    base, quote = pair_currencies[symbol]

                    # Check if we have strength data for both currencies
                    if base in currency_strength_data and quote in currency_strength_data:
                        base_strength = currency_strength_data[base]
                        quote_strength = currency_strength_data[quote]

                        # Calculate expected direction based on relative strength
                        strength_diff = base_strength - quote_strength
                        expected_direction_from_strength = "bullish" if strength_diff > 0 else "bearish"

                        # Check if strength-based direction matches signal direction
                        if expected_direction_from_strength == signal_direction:
                            # Boost confluence score if currency strength confirms signal
                            confluence_score = min(1.0, confluence_score + 0.1)  # 10% boost
                            currency_strength_confirms = True
                        else:
                            # Slight penalty if currency strength contradicts signal
                            confluence_score = max(0.0, confluence_score - 0.05)  # 5% penalty
                            currency_strength_confirms = False
            except Exception as e:
                self.logger.warning(f"Error incorporating currency strength: {e}")

        # Record performance metric
        execution_time = time.time() - start_time
        with self.metrics_lock:
            self.performance_metrics["detect_confluence"].append(execution_time)

        # Log performance details if it took longer than expected
        if execution_time > 0.2:  # Log if > 200ms
            self.logger.debug(
                f"detect_confluence performance: {execution_time:.4f}s, "
                f"analyzed {len(available_related)} pairs, score: {confluence_score:.2f}"
            )

        return {
            "symbol": symbol,
            "signal_type": signal_type,
            "signal_direction": signal_direction,
            "related_pairs_count": total_pairs,
            "confirmation_count": confirmation_count,
            "contradiction_count": contradiction_count,
            "neutral_count": neutral_count,
            "confluence_score": confluence_score,
            "confirmations": confirmations,
            "contradictions": contradictions,
            "neutral": neutral,
            "currency_strength_confirms": currency_strength_confirms,
            "timestamp": datetime.now().isoformat(),
            "execution_time_ms": round(execution_time * 1000, 2)
        }

    def _get_signal_cache_key(self, price_data: pd.DataFrame, signal_type: str) -> str:
        """Generate a cache key for signal detection"""
        # Use the last few rows of data as a fingerprint
        last_rows = min(5, len(price_data))

        if last_rows == 0:
            return f"{signal_type}_empty"

        # Get the close column
        close_col = next((col for col in price_data.columns if col.lower() in ['close', 'price', 'adj close']), None)
        if not close_col:
            return f"{signal_type}_no_close"

        # Create a fingerprint from the last N rows
        close_vals = price_data[close_col].iloc[-last_rows:].values

        # Use hash of the values as the key
        fingerprint = hash(tuple(close_vals))
        return f"{signal_type}_{fingerprint}"

    def _detect_signal(
        self,
        price_data: pd.DataFrame,
        signal_type: str,
        lookback: int
    ) -> Optional[Dict[str, Any]]:
        """
        Detect a specific type of signal in price data.

        Optimized with:
        - Enhanced caching with TTL
        - Early termination for invalid inputs
        - Vectorized operations for faster calculations
        - Performance monitoring
        - Memory optimization

        Args:
            price_data: Price DataFrame for a currency pair
            signal_type: Type of signal to detect
            lookback: Number of periods to look back

        Returns:
            Dictionary with signal details or None if no signal detected
        """
        start_time = time.time()

        # Early termination for invalid inputs
        if price_data is None or price_data.empty or len(price_data) < lookback:
            return None

        # Get the relevant columns
        close_col = next((col for col in price_data.columns if col.lower() in ['close', 'price', 'adj close']), None)
        if not close_col:
            return None

        # Check cache with optimized key generation
        cache_key = self._get_signal_cache_key(price_data, signal_type)
        with self.cache_lock:
            if cache_key in self.signal_cache:
                timestamp, cached_result = self.signal_cache[cache_key]
                if time.time() - timestamp < (self.cache_ttl_minutes * 60):
                    # Return a copy to prevent modification of cached data
                    return cached_result.copy() if cached_result else None

        # Implement different signal detection methods based on signal_type
        # Use try-except to handle potential errors in signal detection
        try:
            result = None
            if signal_type == "trend":
                result = self._detect_trend_signal(price_data, close_col, lookback)
            elif signal_type == "reversal":
                result = self._detect_reversal_signal(price_data, close_col, lookback)
            elif signal_type == "breakout":
                result = self._detect_breakout_signal(price_data, close_col, lookback)
            else:
                self.logger.warning(f"Unknown signal type: {signal_type}")
                return None  # Early return for unknown signal type

            # Early return if no signal detected
            if result is None:
                # Cache the negative result to avoid repeated calculations
                with self.cache_lock:
                    self.signal_cache[cache_key] = (time.time(), None)
                return None

            # Add execution time for performance monitoring
            execution_time = time.time() - start_time
            if execution_time > 0.05:  # Log if > 50ms
                self.logger.debug(
                    f"Signal detection ({signal_type}) took {execution_time:.4f}s"
                )

            # Cache the result with thread safety
            with self.cache_lock:
                self.signal_cache[cache_key] = (time.time(), result)

            return result

        except Exception as e:
            self.logger.error(f"Error detecting {signal_type} signal: {e}")

            # Cache the error result to avoid repeated failures
            with self.cache_lock:
                self.signal_cache[cache_key] = (time.time(), None)

            return None

    def _detect_trend_signal(
        self,
        price_data: pd.DataFrame,
        close_col: str,
        lookback: int
    ) -> Optional[Dict[str, Any]]:
        """
        Detect trend signals in price data.

        Optimized with:
        - Vectorized operations
        - Early termination for invalid inputs
        - Improved strength calculation

        Args:
            price_data: Price DataFrame
            close_col: Name of the close price column
            lookback: Number of periods to look back

        Returns:
            Dictionary with signal details or None if no signal detected
        """
        # Early termination for insufficient data
        if len(price_data) < 50:  # Need at least 50 bars for 50-period MA
            return None

        # Calculate short and medium-term moving averages using vectorized operations
        # Use numpy for faster calculations if possible
        prices = price_data[close_col].values

        # Calculate short MA (20-period)
        short_window = 20
        if len(prices) < short_window:
            return None

        # Use numpy's convolve for faster MA calculation
        short_weights = np.ones(short_window) / short_window
        short_ma_values = np.convolve(prices, short_weights, 'valid')
        short_ma_last = short_ma_values[-1]

        # Calculate medium MA (50-period)
        medium_window = 50
        if len(prices) < medium_window:
            return None

        medium_weights = np.ones(medium_window) / medium_window
        medium_ma_values = np.convolve(prices, medium_weights, 'valid')
        medium_ma_last = medium_ma_values[-1]

        # Check for valid MAs
        if np.isnan(short_ma_last) or np.isnan(medium_ma_last) or medium_ma_last == 0:
            return None

        # Determine trend direction and calculate strength
        ma_ratio = short_ma_last / medium_ma_last

        if ma_ratio > 1.0:
            direction = "bullish"
            # Calculate strength based on how far short MA is above medium MA
            # Use sigmoid-like function for more nuanced strength calculation
            strength = min(1.0, 2 / (1 + np.exp(-10 * (ma_ratio - 1))) - 1)
        elif ma_ratio < 1.0:
            direction = "bearish"
            # Calculate strength based on how far short MA is below medium MA
            # Use sigmoid-like function for more nuanced strength calculation
            strength = min(1.0, 2 / (1 + np.exp(-10 * (1 - ma_ratio))) - 1)
        else:
            return None  # No clear trend

        # Calculate additional trend metrics for more comprehensive analysis
        # Trend consistency: how consistently the short MA has been above/below medium MA
        if len(short_ma_values) > 5 and len(medium_ma_values) > 5:
            # Get the last 5 values of each MA
            short_recent = short_ma_values[-5:]
            medium_recent = medium_ma_values[-5:]

            # Calculate how many periods the trend has been consistent
            if direction == "bullish":
                consistency = np.sum(short_recent > medium_recent) / 5
            else:
                consistency = np.sum(short_recent < medium_recent) / 5
        else:
            consistency = 1.0  # Default if we don't have enough data

        return {
            "type": "trend",
            "direction": direction,
            "strength": float(strength),  # Convert numpy types to Python types
            "consistency": float(consistency),
            "short_ma": float(short_ma_last),
            "medium_ma": float(medium_ma_last),
            "ma_ratio": float(ma_ratio)
        }

    def _detect_reversal_signal(
        self,
        price_data: pd.DataFrame,
        close_col: str,
        lookback: int
    ) -> Optional[Dict[str, Any]]:
        """
        Detect reversal signals in price data.

        Args:
            price_data: Price DataFrame
            close_col: Name of the close price column
            lookback: Number of periods to look back

        Returns:
            Dictionary with signal details or None if no signal detected
        """
        # Calculate RSI if not already in the data
        rsi_col = next((col for col in price_data.columns if col.lower() == 'rsi'), None)

        if rsi_col:
            rsi = price_data[rsi_col].iloc[-1]
        else:
            # Calculate RSI
            delta = price_data[close_col].diff()
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = abs(loss)

            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.iloc[-1]

        # Check for reversal conditions
        if rsi < 30:  # Oversold
            direction = "bullish"  # Potential bullish reversal
            strength = min(1.0, (30 - rsi) / 30)  # Lower RSI = stronger signal
        elif rsi > 70:  # Overbought
            direction = "bearish"  # Potential bearish reversal
            strength = min(1.0, (rsi - 70) / 30)  # Higher RSI = stronger signal
        else:
            return None  # No reversal signal

        return {
            "type": "reversal",
            "direction": direction,
            "strength": strength,
            "rsi": rsi
        }

    def _detect_breakout_signal(
        self,
        price_data: pd.DataFrame,
        close_col: str,
        lookback: int
    ) -> Optional[Dict[str, Any]]:
        """
        Detect breakout signals in price data.

        Args:
            price_data: Price DataFrame
            close_col: Name of the close price column
            lookback: Number of periods to look back

        Returns:
            Dictionary with signal details or None if no signal detected
        """
        if len(price_data) < lookback:
            return None

        # Calculate recent high and low
        recent_data = price_data.iloc[-lookback:]
        recent_high = recent_data[close_col].max()
        recent_low = recent_data[close_col].min()

        # Get current price
        current_price = price_data[close_col].iloc[-1]

        # Calculate price range
        price_range = recent_high - recent_low
        if price_range == 0:
            return None  # Avoid division by zero

        # Check for breakout
        if current_price > recent_high * 0.99:  # Within 1% of recent high
            direction = "bullish"
            # Calculate strength based on how far price is above recent high
            strength = min(1.0, (current_price / recent_high - 1) * 10)
        elif current_price < recent_low * 1.01:  # Within 1% of recent low
            direction = "bearish"
            # Calculate strength based on how far price is below recent low
            strength = min(1.0, (1 - current_price / recent_low) * 10)
        else:
            return None  # No breakout

        return {
            "type": "breakout",
            "direction": direction,
            "strength": strength,
            "recent_high": recent_high,
            "recent_low": recent_low
        }

    def analyze_divergence(
        self,
        symbol: str,
        price_data: Dict[str, pd.DataFrame],
        related_pairs: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze divergence between related pairs, which can indicate potential reversals.

        Optimized with:
        - Enhanced caching for momentum calculations
        - Parallel processing for analyzing multiple pairs
        - Early termination for invalid inputs
        - Vectorized operations where possible

        Args:
            symbol: The primary currency pair
            price_data: Dictionary mapping currency pairs to price DataFrames
            related_pairs: Optional dictionary of related pairs (if not provided, will be calculated)

        Returns:
            Dictionary with divergence analysis results
        """
        start_time = time.time()

        # Early termination for invalid inputs
        if symbol not in price_data or price_data[symbol].empty:
            return {"error": f"No price data available for {symbol}"}

        # Get related pairs if not provided
        if related_pairs is None:
            # Since this is a synchronous method, we can't use the async find_related_pairs
            related_pairs = {}

        # Filter price data to include only related pairs that we have data for
        available_related = {
            pair: corr for pair, corr in related_pairs.items()
            if pair in price_data and not price_data[pair].empty
        }

        # Early termination if no related pairs
        if not available_related:
            return {
                "symbol": symbol,
                "divergences_found": 0,
                "message": "No related pairs with available data"
            }

        # Calculate momentum for primary pair with caching
        primary_momentum = self._calculate_momentum_cached(price_data[symbol])

        # Early termination if primary momentum calculation failed
        if primary_momentum is None:
            return {
                "symbol": symbol,
                "divergences_found": 0,
                "message": "Could not calculate momentum for primary pair"
            }

        # Use parallel processing for analyzing multiple pairs
        divergences = []

        # Use ThreadPoolExecutor for parallel processing if we have enough pairs
        if len(available_related) >= 3 and self.max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(available_related))) as executor:
                # Create tasks for each related pair
                future_to_pair = {
                    executor.submit(
                        self._analyze_pair_divergence,
                        related_pair,
                        correlation,
                        price_data[related_pair],
                        primary_momentum
                    ): related_pair
                    for related_pair, correlation in available_related.items()
                }

                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_pair):
                    result = future.result()
                    if result:
                        divergences.append(result)
        else:
            # Sequential processing for small number of pairs
            for related_pair, correlation in available_related.items():
                result = self._analyze_pair_divergence(
                    related_pair,
                    correlation,
                    price_data[related_pair],
                    primary_momentum
                )
                if result:
                    divergences.append(result)

        # Record performance metric
        execution_time = time.time() - start_time
        with self.metrics_lock:
            self.performance_metrics["analyze_divergence"].append(execution_time)

        # Log performance details if it took longer than expected
        if execution_time > 0.1:  # Log if > 100ms
            self.logger.debug(
                f"analyze_divergence performance: {execution_time:.4f}s, "
                f"analyzed {len(available_related)} pairs, found {len(divergences)} divergences"
            )

        return {
            "symbol": symbol,
            "primary_momentum": primary_momentum,
            "divergences_found": len(divergences),
            "divergences": divergences,
            "execution_time_ms": round(execution_time * 1000, 2)
        }

    def _analyze_pair_divergence(
        self,
        related_pair: str,
        correlation: float,
        price_data: pd.DataFrame,
        primary_momentum: float
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze divergence for a single related pair.

        Args:
            related_pair: The related currency pair
            correlation: Correlation value with primary pair
            price_data: Price DataFrame for the related pair
            primary_momentum: Momentum value for the primary pair

        Returns:
            Dictionary with divergence details or None if no divergence
        """
        # Calculate momentum for related pair with caching
        related_momentum = self._calculate_momentum_cached(price_data)

        # Skip if momentum calculation failed
        if related_momentum is None:
            return None

        # Determine expected momentum based on correlation
        expected_momentum = primary_momentum
        if correlation < 0:
            # For negative correlation, expect opposite momentum
            expected_momentum = -primary_momentum

        # Check for divergence
        momentum_diff = related_momentum - expected_momentum

        # Significant divergence if difference exceeds threshold
        divergence_threshold = 0.5  # Threshold for significant divergence
        if abs(momentum_diff) > divergence_threshold:
            return {
                "pair": related_pair,
                "correlation": correlation,
                "primary_momentum": primary_momentum,
                "related_momentum": related_momentum,
                "expected_momentum": expected_momentum,
                "momentum_difference": momentum_diff,
                "divergence_type": "positive" if momentum_diff > 0 else "negative",
                "divergence_strength": min(1.0, abs(momentum_diff) / 2.0)  # Normalize strength to 0-1 range
            }

        return None

    def _calculate_momentum_cached(self, price_data: pd.DataFrame) -> Optional[float]:
        """
        Calculate momentum for a price series with caching.

        Args:
            price_data: Price DataFrame

        Returns:
            Momentum value (-1.0 to 1.0) or None if calculation fails
        """
        # Early termination for invalid inputs
        if price_data is None or price_data.empty or len(price_data) < 14:
            return None

        # Get the close column
        close_col = next((col for col in price_data.columns if col.lower() in ['close', 'price', 'adj close']), None)
        if not close_col:
            return None

        # Create cache key
        # Use the last few rows of data as a fingerprint
        last_rows = min(5, len(price_data))
        close_vals = price_data[close_col].iloc[-last_rows:].values
        cache_key = f"momentum_{hash(tuple(close_vals))}"

        # Check cache
        with self.cache_lock:
            if cache_key in self.momentum_cache:
                timestamp, cached_result = self.momentum_cache[cache_key]
                if time.time() - timestamp < (self.cache_ttl_minutes * 60):
                    return cached_result

        # Calculate momentum if not in cache
        momentum = self._calculate_momentum(price_data)

        # Update cache
        with self.cache_lock:
            self.momentum_cache[cache_key] = (time.time(), momentum)

        return momentum

    def _calculate_momentum(self, price_data: pd.DataFrame) -> float:
        """
        Calculate momentum for a price series.

        Optimized with:
        - Vectorized operations for faster calculation
        - Early termination for invalid inputs
        - Multiple momentum indicators for more robust calculation

        Args:
            price_data: Price DataFrame

        Returns:
            Momentum value (-1.0 to 1.0)
        """
        close_col = next((col for col in price_data.columns if col.lower() in ['close', 'price', 'adj close']), None)
        if not close_col or len(price_data) < 14:
            return 0.0

        # Get close prices as numpy array for faster calculations
        close_prices = price_data[close_col].values

        # Early termination if we have invalid data
        if len(close_prices) == 0 or np.isnan(close_prices[-1]):
            return 0.0

        # Calculate multiple momentum indicators for a more robust measure

        # 1. Rate of Change (ROC)
        roc_period = 10
        if len(close_prices) <= roc_period:
            return 0.0

        current_price = close_prices[-1]
        past_price = close_prices[-roc_period-1]

        if past_price == 0 or np.isnan(past_price):
            roc_momentum = 0.0
        else:
            roc = (current_price / past_price - 1) * 100
            # Normalize to -1.0 to 1.0 range
            roc_momentum = max(-1.0, min(1.0, roc / 10.0))

        # 2. Moving Average Convergence/Divergence (MACD) momentum
        if len(close_prices) >= 26:
            # Calculate EMAs using numpy for better performance
            ema12 = self._calculate_ema(close_prices, 12)
            ema26 = self._calculate_ema(close_prices, 26)

            # Calculate MACD line
            macd_line = ema12 - ema26

            # Calculate signal line (9-period EMA of MACD line)
            signal_line = self._calculate_ema(macd_line, 9)

            # Calculate histogram
            histogram = macd_line - signal_line

            # Use histogram direction and magnitude for momentum
            if len(histogram) >= 2:
                macd_momentum = max(-1.0, min(1.0, histogram[-1] / 0.001))  # Normalize
            else:
                macd_momentum = 0.0
        else:
            macd_momentum = 0.0

        # 3. RSI directional momentum
        if len(close_prices) >= 14:
            # Calculate price changes
            delta = np.diff(close_prices)

            # Separate gains and losses
            gains = np.copy(delta)
            losses = np.copy(delta)
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = np.abs(losses)

            # Calculate average gains and losses
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])

            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            # Convert RSI to momentum (-1.0 to 1.0)
            # RSI of 50 = neutral (0), RSI of 100 = strong momentum (1), RSI of 0 = strong negative momentum (-1)
            rsi_momentum = (rsi - 50) / 50
        else:
            rsi_momentum = 0.0

        # Combine all momentum indicators with weights
        combined_momentum = (roc_momentum * 0.5) + (macd_momentum * 0.3) + (rsi_momentum * 0.2)

        # Ensure result is in -1.0 to 1.0 range
        return max(-1.0, min(1.0, combined_momentum))

    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average using vectorized operations.

        Args:
            data: Price data as numpy array
            period: EMA period

        Returns:
            EMA values as numpy array
        """
        if len(data) < period:
            return np.array([])

        # Calculate multiplier
        multiplier = 2 / (period + 1)

        # Initialize EMA with SMA
        ema = np.zeros_like(data)
        ema[:period] = np.mean(data[:period])

        # Calculate EMA
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]

        return ema

    def _extract_currencies_from_pairs(self, pairs: List[str]) -> Dict[str, Tuple[str, str]]:
        """
        Extract base and quote currencies from a list of currency pairs.

        Args:
            pairs: List of currency pair symbols (e.g., ["EUR/USD", "GBP/JPY"])

        Returns:
            Dictionary mapping pairs to (base, quote) tuples
        """
        result = {}

        for pair in pairs:
            # Handle different separator formats
            if "/" in pair:
                base, quote = pair.split("/")
            elif "_" in pair:
                base, quote = pair.split("_")
            else:
                # Try to split based on common currency codes
                # This is a simplified approach and might not work for all pairs
                common_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]

                found = False
                for curr in common_currencies:
                    if pair.endswith(curr):
                        base = pair[:-len(curr)]
                        quote = curr
                        found = True
                        break
                    elif pair.startswith(curr):
                        base = curr
                        quote = pair[len(curr):]
                        found = True
                        break

                if not found:
                    # Default to splitting in the middle for 6-character pairs
                    if len(pair) == 6:
                        base = pair[:3]
                        quote = pair[3:]
                    else:
                        self.logger.warning(f"Could not parse currency pair: {pair}")
                        continue

            result[pair] = (base, quote)

        return result
