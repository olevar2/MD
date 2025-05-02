"""
Related Pairs Confluence Detector Module

This module provides functionality for detecting confluence signals across related currency pairs,
which can strengthen trading signals and improve prediction accuracy.

Part of Phase 2 implementation to enhance currency correlation analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta

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
        lookback_periods: int = 20
    ):
        """
        Initialize the related pairs confluence detector.

        Args:
            correlation_service: Service for tracking correlations between pairs
            currency_strength_analyzer: Service for analyzing currency strength
            correlation_threshold: Minimum correlation to consider pairs related
            lookback_periods: Number of periods to look back for analysis
        """
        self.correlation_service = correlation_service
        self.currency_strength_analyzer = currency_strength_analyzer
        self.correlation_threshold = correlation_threshold
        self.lookback_periods = lookback_periods

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Cache for related pairs
        self.related_pairs_cache = {}
        self.cache_expiry = datetime.now()

        self.logger.info(f"RelatedPairsConfluenceAnalyzer initialized with correlation threshold {correlation_threshold}")

    async def find_related_pairs(self, symbol: str, min_correlation: Optional[float] = None) -> Dict[str, float]:
        """
        Find pairs that are correlated with the given symbol.

        Args:
            symbol: The currency pair to find related pairs for
            min_correlation: Optional minimum correlation threshold (overrides default)

        Returns:
            Dictionary mapping related pairs to their correlation values
        """
        if not self.correlation_service:
            self.logger.warning("No correlation service provided, cannot find related pairs")
            return {}

        # Use provided threshold or default
        threshold = min_correlation if min_correlation is not None else self.correlation_threshold

        # Check cache
        cache_key = f"{symbol}_{threshold}"
        if cache_key in self.related_pairs_cache and datetime.now() < self.cache_expiry:
            return self.related_pairs_cache[cache_key]

        # Get correlation matrix from service
        try:
            # This assumes the correlation service has a method to get all correlations
            all_correlations = await self.correlation_service.get_all_correlations()

            # Filter for correlations with our symbol
            related = {}
            for pair, correlations in all_correlations.items():
                if symbol in correlations:
                    corr_value = correlations[symbol]
                    if abs(corr_value) >= threshold:
                        related[pair] = corr_value

            # Update cache
            self.related_pairs_cache[cache_key] = related
            self.cache_expiry = datetime.now() + timedelta(hours=1)  # Cache for 1 hour

            return related

        except Exception as e:
            self.logger.error(f"Error finding related pairs: {e}")
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
        if use_currency_strength and self.currency_strength_analyzer:
            try:
                # Calculate currency strength
                currency_strength_data = self.currency_strength_analyzer.calculate_currency_strength(price_data)

                # Get divergence signals if available
                divergence_signals = self.currency_strength_analyzer.compute_divergence_signals(price_data)
            except Exception as e:
                self.logger.warning(f"Error calculating currency strength: {e}")

        # Analyze each related pair for the same signal type and direction
        confirmations = []
        contradictions = []
        neutral = []

        for related_pair, correlation in available_related.items():
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
                neutral.append({
                    "pair": related_pair,
                    "correlation": correlation,
                    "reason": "no_signal_detected"
                })
                continue

            # Check if direction matches expected and signal is strong enough
            if related_signal["direction"] == expected_direction and related_signal.get("strength", 0.0) >= min_confirmation_strength:
                confirmations.append({
                    "pair": related_pair,
                    "correlation": correlation,
                    "signal_strength": related_signal.get("strength", 0.0),
                    "expected_direction": expected_direction,
                    "signal_type": related_signal.get("type", "unknown")
                })
            elif related_signal.get("strength", 0.0) >= min_confirmation_strength:
                contradictions.append({
                    "pair": related_pair,
                    "correlation": correlation,
                    "signal_strength": related_signal.get("strength", 0.0),
                    "expected_direction": expected_direction,
                    "actual_direction": related_signal["direction"],
                    "signal_type": related_signal.get("type", "unknown")
                })
            else:
                # Signal too weak to be considered
                neutral.append({
                    "pair": related_pair,
                    "correlation": correlation,
                    "signal_direction": related_signal["direction"],
                    "signal_strength": related_signal.get("strength", 0.0),
                    "reason": "weak_signal"
                })

        # Calculate confluence score
        total_pairs = len(available_related)
        confirmation_count = len(confirmations)
        contradiction_count = len(contradictions)
        neutral_count = len(neutral)

        # Weight confirmations by their correlation and signal strength
        weighted_confirmations = sum(
            abs(conf["correlation"]) * conf["signal_strength"]
            for conf in confirmations
        )

        # Weight contradictions by their correlation and signal strength
        weighted_contradictions = sum(
            abs(cont["correlation"]) * cont["signal_strength"]
            for cont in contradictions
        )

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
            "timestamp": datetime.now().isoformat()
        }

    def _detect_signal(
        self,
        price_data: pd.DataFrame,
        signal_type: str,
        lookback: int
    ) -> Optional[Dict[str, Any]]:
        """
        Detect a specific type of signal in price data.

        Args:
            price_data: Price DataFrame for a currency pair
            signal_type: Type of signal to detect
            lookback: Number of periods to look back

        Returns:
            Dictionary with signal details or None if no signal detected
        """
        if price_data.empty or len(price_data) < lookback:
            return None

        # Get the relevant columns
        close_col = next((col for col in price_data.columns if col.lower() in ['close', 'price', 'adj close']), None)
        if not close_col:
            return None

        # Get the most recent close price
        current_price = price_data[close_col].iloc[-1]

        # Implement different signal detection methods based on signal_type
        if signal_type == "trend":
            return self._detect_trend_signal(price_data, close_col, lookback)
        elif signal_type == "reversal":
            return self._detect_reversal_signal(price_data, close_col, lookback)
        elif signal_type == "breakout":
            return self._detect_breakout_signal(price_data, close_col, lookback)
        else:
            self.logger.warning(f"Unknown signal type: {signal_type}")
            return None

    def _detect_trend_signal(
        self,
        price_data: pd.DataFrame,
        close_col: str,
        lookback: int
    ) -> Optional[Dict[str, Any]]:
        """
        Detect trend signals in price data.

        Args:
            price_data: Price DataFrame
            close_col: Name of the close price column
            lookback: Number of periods to look back

        Returns:
            Dictionary with signal details or None if no signal detected
        """
        # Calculate short and medium-term moving averages
        short_ma = price_data[close_col].rolling(window=20).mean()
        medium_ma = price_data[close_col].rolling(window=50).mean()

        # Check for valid MAs
        if short_ma.iloc[-1] is np.nan or medium_ma.iloc[-1] is np.nan:
            return None

        # Determine trend direction
        if short_ma.iloc[-1] > medium_ma.iloc[-1]:
            direction = "bullish"
            # Calculate strength based on how far short MA is above medium MA
            strength = min(1.0, (short_ma.iloc[-1] / medium_ma.iloc[-1] - 1) * 10)
        elif short_ma.iloc[-1] < medium_ma.iloc[-1]:
            direction = "bearish"
            # Calculate strength based on how far short MA is below medium MA
            strength = min(1.0, (1 - short_ma.iloc[-1] / medium_ma.iloc[-1]) * 10)
        else:
            return None  # No clear trend

        return {
            "type": "trend",
            "direction": direction,
            "strength": strength,
            "short_ma": short_ma.iloc[-1],
            "medium_ma": medium_ma.iloc[-1]
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

        Args:
            symbol: The primary currency pair
            price_data: Dictionary mapping currency pairs to price DataFrames
            related_pairs: Optional dictionary of related pairs (if not provided, will be calculated)

        Returns:
            Dictionary with divergence analysis results
        """
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

        if not available_related:
            return {
                "symbol": symbol,
                "divergences_found": 0,
                "message": "No related pairs with available data"
            }

        # Calculate momentum for primary pair
        primary_momentum = self._calculate_momentum(price_data[symbol])

        # Analyze each related pair for divergence
        divergences = []

        for related_pair, correlation in available_related.items():
            # Calculate momentum for related pair
            related_momentum = self._calculate_momentum(price_data[related_pair])

            # Determine expected momentum based on correlation
            expected_momentum = primary_momentum
            if correlation < 0:
                # For negative correlation, expect opposite momentum
                expected_momentum = -primary_momentum

            # Check for divergence
            momentum_diff = related_momentum - expected_momentum

            # Significant divergence if difference exceeds threshold
            if abs(momentum_diff) > 0.5:  # Threshold for significant divergence
                divergences.append({
                    "pair": related_pair,
                    "correlation": correlation,
                    "primary_momentum": primary_momentum,
                    "related_momentum": related_momentum,
                    "expected_momentum": expected_momentum,
                    "momentum_difference": momentum_diff,
                    "divergence_type": "positive" if momentum_diff > 0 else "negative"
                })

        return {
            "symbol": symbol,
            "primary_momentum": primary_momentum,
            "divergences_found": len(divergences),
            "divergences": divergences
        }

    def _calculate_momentum(self, price_data: pd.DataFrame) -> float:
        """
        Calculate momentum for a price series.

        Args:
            price_data: Price DataFrame

        Returns:
            Momentum value (-1.0 to 1.0)
        """
        close_col = next((col for col in price_data.columns if col.lower() in ['close', 'price', 'adj close']), None)
        if not close_col or len(price_data) < 14:
            return 0.0

        # Calculate rate of change
        roc_period = 10
        current_price = price_data[close_col].iloc[-1]
        past_price = price_data[close_col].iloc[-roc_period-1]

        if past_price == 0:
            return 0.0

        roc = (current_price / past_price - 1) * 100

        # Normalize to -1.0 to 1.0 range
        normalized_roc = max(-1.0, min(1.0, roc / 10.0))

        return normalized_roc

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
