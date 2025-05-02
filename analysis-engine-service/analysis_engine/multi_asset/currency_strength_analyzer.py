"""
Currency Strength Analyzer Module

This module provides functionality for calculating and analyzing currency strength
across multiple currency pairs. It helps identify strong and weak currencies,
which can be used to find trading opportunities.

Part of Phase 2 implementation to enhance currency correlation analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict

from analysis_engine.multi_asset.correlation_tracking_service import CorrelationTrackingService


class CurrencyStrengthAnalyzer:
    """
    Analyzes the relative strength of individual currencies based on the performance
    of currency pairs they are part of.

    This service:
    - Calculates individual currency strength metrics
    - Tracks currency strength over time
    - Identifies divergence/convergence between related pairs
    - Provides currency basket analysis capabilities
    """

    def __init__(
        self,
        base_currencies: Optional[List[str]] = None,
        quote_currencies: Optional[List[str]] = None,
        lookback_periods: int = 20,
        correlation_service: Optional[CorrelationTrackingService] = None
    ):
        """
        Initialize the currency strength analyzer.

        Args:
            base_currencies: List of base currencies to track (e.g., ["EUR", "GBP", "AUD"])
            quote_currencies: List of quote currencies to track (e.g., ["USD", "JPY", "CHF"])
            lookback_periods: Number of periods to use for strength calculation
            correlation_service: Optional correlation service for related pair analysis
        """
        # Default major currencies if not provided
        self.base_currencies = base_currencies or ["EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF", "JPY"]
        self.quote_currencies = quote_currencies or ["USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD"]
        self.lookback_periods = lookback_periods
        self.correlation_service = correlation_service

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Initialize strength history
        self.strength_history = {}
        self.last_calculation_time = None

        self.logger.info(f"CurrencyStrengthAnalyzer initialized with {len(self.base_currencies)} currencies")

    def calculate_currency_strength(
        self,
        price_data: Dict[str, pd.DataFrame],
        method: str = "price_change",
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Calculate the strength of individual currencies based on price data.

        Args:
            price_data: Dictionary mapping currency pairs to price DataFrames
            method: Method to use for strength calculation:
                    - "price_change": Based on price change percentage
                    - "momentum": Based on momentum indicators
                    - "rsi": Based on RSI values
            normalize: Whether to normalize strength values to [-1, 1] range

        Returns:
            Dictionary mapping currency codes to strength values
        """
        if not price_data:
            self.logger.warning("No price data provided for currency strength calculation")
            return {}

        # Extract all unique currencies from the pairs
        all_currencies = set()
        pair_currencies = {}

        for pair, df in price_data.items():
            if df.empty:
                continue

            # Parse currency pair (assuming format like "EUR/USD")
            parts = pair.split('/')
            if len(parts) != 2:
                self.logger.warning(f"Invalid currency pair format: {pair}")
                continue

            base, quote = parts
            all_currencies.add(base)
            all_currencies.add(quote)
            pair_currencies[pair] = (base, quote)

        # Initialize strength values
        raw_strength = {currency: [] for currency in all_currencies}

        # Calculate strength based on selected method
        if method == "price_change":
            strength_values = self._calculate_strength_by_price_change(price_data, pair_currencies)
        elif method == "momentum":
            strength_values = self._calculate_strength_by_momentum(price_data, pair_currencies)
        elif method == "rsi":
            strength_values = self._calculate_strength_by_rsi(price_data, pair_currencies)
        else:
            self.logger.warning(f"Unknown strength calculation method: {method}")
            return {}

        # Normalize if requested
        if normalize and strength_values:
            max_abs = max(abs(v) for v in strength_values.values())
            if max_abs > 0:
                strength_values = {k: v / max_abs for k, v in strength_values.items()}

        # Update history
        self._update_strength_history(strength_values)

        return strength_values

    def _calculate_strength_by_price_change(
        self,
        price_data: Dict[str, pd.DataFrame],
        pair_currencies: Dict[str, Tuple[str, str]]
    ) -> Dict[str, float]:
        """
        Calculate currency strength based on price change percentage.

        Args:
            price_data: Dictionary mapping currency pairs to price DataFrames
            pair_currencies: Dictionary mapping pairs to (base, quote) tuples

        Returns:
            Dictionary mapping currency codes to strength values
        """
        # Initialize contribution counters and strength values
        contributions = {currency: 0 for currency in set(c for pair in pair_currencies.values() for c in pair)}
        strength_values = {currency: 0.0 for currency in contributions}

        for pair, df in price_data.items():
            if pair not in pair_currencies or df.empty:
                continue

            base, quote = pair_currencies[pair]

            # Calculate price change over lookback period
            if len(df) >= self.lookback_periods:
                close_col = next((col for col in df.columns if col.lower() in ['close', 'price', 'adj close']), None)
                if not close_col:
                    continue

                start_price = df[close_col].iloc[-self.lookback_periods]
                end_price = df[close_col].iloc[-1]

                if start_price > 0:
                    pct_change = (end_price - start_price) / start_price

                    # Positive change means base currency strengthened relative to quote
                    strength_values[base] += pct_change
                    strength_values[quote] -= pct_change

                    contributions[base] += 1
                    contributions[quote] += 1

        # Average the strength values by the number of contributions
        for currency in strength_values:
            if contributions[currency] > 0:
                strength_values[currency] /= contributions[currency]

        return strength_values

    def _calculate_strength_by_momentum(
        self,
        price_data: Dict[str, pd.DataFrame],
        pair_currencies: Dict[str, Tuple[str, str]]
    ) -> Dict[str, float]:
        """
        Calculate currency strength based on momentum indicators.

        Args:
            price_data: Dictionary mapping currency pairs to price DataFrames
            pair_currencies: Dictionary mapping pairs to (base, quote) tuples

        Returns:
            Dictionary mapping currency codes to strength values
        """
        # Initialize contribution counters and strength values
        contributions = {currency: 0 for currency in set(c for pair in pair_currencies.values() for c in pair)}
        strength_values = {currency: 0.0 for currency in contributions}

        for pair, df in price_data.items():
            if pair not in pair_currencies or df.empty or len(df) < self.lookback_periods:
                continue

            base, quote = pair_currencies[pair]

            # Calculate momentum (rate of change)
            close_col = next((col for col in df.columns if col.lower() in ['close', 'price', 'adj close']), None)
            if not close_col:
                continue

            # Calculate 14-period ROC
            roc_period = min(14, len(df) - 1)
            roc = (df[close_col].iloc[-1] / df[close_col].iloc[-roc_period-1] - 1) * 100

            # Positive ROC means base currency strengthened relative to quote
            strength_values[base] += roc
            strength_values[quote] -= roc

            contributions[base] += 1
            contributions[quote] += 1

        # Average the strength values by the number of contributions
        for currency in strength_values:
            if contributions[currency] > 0:
                strength_values[currency] /= contributions[currency]

        return strength_values

    def _calculate_strength_by_rsi(
        self,
        price_data: Dict[str, pd.DataFrame],
        pair_currencies: Dict[str, Tuple[str, str]]
    ) -> Dict[str, float]:
        """
        Calculate currency strength based on RSI values.

        Args:
            price_data: Dictionary mapping currency pairs to price DataFrames
            pair_currencies: Dictionary mapping pairs to (base, quote) tuples

        Returns:
            Dictionary mapping currency codes to strength values
        """
        # Initialize contribution counters and strength values
        contributions = {currency: 0 for currency in set(c for pair in pair_currencies.values() for c in pair)}
        strength_values = {currency: 0.0 for currency in contributions}

        for pair, df in price_data.items():
            if pair not in pair_currencies or df.empty or len(df) < self.lookback_periods:
                continue

            base, quote = pair_currencies[pair]

            # Check if RSI is already calculated
            rsi_col = next((col for col in df.columns if col.lower() == 'rsi'), None)

            if rsi_col:
                rsi = df[rsi_col].iloc[-1]
            else:
                # Calculate RSI if not available
                close_col = next((col for col in df.columns if col.lower() in ['close', 'price', 'adj close']), None)
                if not close_col:
                    continue

                # Calculate price changes
                price_changes = df[close_col].diff()

                # Calculate gains and losses
                gains = price_changes.copy()
                gains[gains < 0] = 0
                losses = -price_changes.copy()
                losses[losses < 0] = 0

                # Calculate average gains and losses
                period = 14
                avg_gain = gains.rolling(window=period).mean().iloc[-1]
                avg_loss = losses.rolling(window=period).mean().iloc[-1]

                # Calculate RSI
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))

            # Convert RSI (0-100) to strength (-1 to 1)
            # RSI > 50 means bullish (base strengthening), < 50 means bearish
            rsi_strength = (rsi - 50) / 50  # Now ranges from -1 to 1

            # Apply to currencies
            strength_values[base] += rsi_strength
            strength_values[quote] -= rsi_strength

            contributions[base] += 1
            contributions[quote] += 1

        # Average the strength values by the number of contributions
        for currency in strength_values:
            if contributions[currency] > 0:
                strength_values[currency] /= contributions[currency]

        return strength_values

    def _update_strength_history(self, strength_values: Dict[str, float]) -> None:
        """
        Update the history of currency strength values.

        Args:
            strength_values: Dictionary mapping currency codes to strength values
        """
        timestamp = datetime.now()

        for currency, strength in strength_values.items():
            if currency not in self.strength_history:
                self.strength_history[currency] = []

            # Add new data point
            self.strength_history[currency].append({
                "timestamp": timestamp,
                "strength": strength
            })

            # Limit history size (keep last 100 points)
            if len(self.strength_history[currency]) > 100:
                self.strength_history[currency] = self.strength_history[currency][-100:]

        self.last_calculation_time = timestamp

    def get_currency_strength_history(
        self,
        currency: str,
        lookback_hours: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the strength history for a specific currency.

        Args:
            currency: Currency code
            lookback_hours: Optional number of hours to look back

        Returns:
            List of strength history data points
        """
        if currency not in self.strength_history:
            return []

        history = self.strength_history[currency]

        if lookback_hours is not None:
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            history = [point for point in history if point["timestamp"] >= cutoff_time]

        return history

    def get_strongest_currencies(self, count: int = 3) -> List[Tuple[str, float]]:
        """
        Get the strongest currencies based on the latest strength values.

        Args:
            count: Number of currencies to return

        Returns:
            List of (currency, strength) tuples, sorted by strength (descending)
        """
        if not self.strength_history:
            return []

        # Get latest strength values for each currency
        latest_strength = {}
        for currency, history in self.strength_history.items():
            if history:
                latest_strength[currency] = history[-1]["strength"]

        # Sort by strength (descending)
        sorted_currencies = sorted(
            latest_strength.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_currencies[:count]

    def get_weakest_currencies(self, count: int = 3) -> List[Tuple[str, float]]:
        """
        Get the weakest currencies based on the latest strength values.

        Args:
            count: Number of currencies to return

        Returns:
            List of (currency, strength) tuples, sorted by strength (ascending)
        """
        if not self.strength_history:
            return []

        # Get latest strength values for each currency
        latest_strength = {}
        for currency, history in self.strength_history.items():
            if history:
                latest_strength[currency] = history[-1]["strength"]

        # Sort by strength (ascending)
        sorted_currencies = sorted(
            latest_strength.items(),
            key=lambda x: x[1]
        )

        return sorted_currencies[:count]

    def find_currency_divergence(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Find currencies with significant divergence from their historical strength.

        Args:
            threshold: Threshold for significant divergence

        Returns:
            List of currencies with significant divergence
        """
        divergences = []

        for currency, history in self.strength_history.items():
            if len(history) < 5:  # Need at least 5 data points
                continue

            # Get current and historical strength
            current_strength = history[-1]["strength"]

            # Calculate average of previous 5-10 points (excluding most recent)
            if len(history) >= 10:
                historical_points = history[-10:-1]
            else:
                historical_points = history[:-1]

            historical_strength = sum(point["strength"] for point in historical_points) / len(historical_points)

            # Calculate divergence
            divergence = current_strength - historical_strength

            # Check if divergence exceeds threshold
            if abs(divergence) >= threshold:
                divergences.append({
                    "currency": currency,
                    "current_strength": current_strength,
                    "historical_strength": historical_strength,
                    "divergence": divergence,
                    "direction": "strengthening" if divergence > 0 else "weakening"
                })

        # Sort by absolute divergence (descending)
        return sorted(divergences, key=lambda x: abs(x["divergence"]), reverse=True)

    def compute_divergence_signals(self, price_data: Dict[str, pd.DataFrame],
                                  lookback_periods: Optional[int] = None,
                                  threshold: float = 0.5) -> Dict[str, Any]:
        """
        Compute divergence and convergence signals across currency pairs.

        This method identifies situations where:
        1. Currency strength diverges from price action (potential reversal signal)
        2. Related pairs show divergent price movements (potential arbitrage or reversal)
        3. Currency baskets show internal divergence (sector rotation signals)

        Args:
            price_data: Dictionary mapping currency pairs to price DataFrames
            lookback_periods: Optional override for lookback periods
            threshold: Threshold for significant divergence

        Returns:
            Dictionary with divergence/convergence signals and analysis
        """
        # Use instance lookback or override
        periods = lookback_periods or self.lookback_periods

        # Calculate current currency strength
        current_strength = self.calculate_currency_strength(price_data)

        # Extract currency pairs from price data
        pairs = list(price_data.keys())
        pair_currencies = self._extract_currencies_from_pairs(pairs)

        # Initialize results
        results = {
            "currency_divergence": [],  # Individual currency divergence
            "pair_divergence": [],     # Price vs strength divergence
            "cross_pair_divergence": [], # Related pairs diverging from each other
            "basket_divergence": [],   # Divergence within currency baskets
            "timestamp": datetime.now().isoformat()
        }

        # 1. Find individual currency divergence from historical strength
        results["currency_divergence"] = self.find_currency_divergence(threshold)

        # 2. Find divergence between price action and currency strength
        for pair, df in price_data.items():
            if pair not in pair_currencies or df.empty or len(df) < periods:
                continue

            base, quote = pair_currencies[pair]

            # Skip if we don't have strength data for both currencies
            if base not in current_strength or quote not in current_strength:
                continue

            # Calculate price change
            close_col = next((col for col in df.columns if col.lower() in ['close', 'price', 'adj close']), None)
            if not close_col:
                continue

            price_change = (df[close_col].iloc[-1] / df[close_col].iloc[-periods] - 1) * 100

            # Calculate expected price change based on relative currency strength
            strength_diff = current_strength[base] - current_strength[quote]
            expected_direction = 1 if strength_diff > 0 else -1
            actual_direction = 1 if price_change > 0 else -1

            # Check for divergence between price action and currency strength
            if expected_direction != actual_direction and abs(strength_diff) >= threshold:
                results["pair_divergence"].append({
                    "pair": pair,
                    "price_change_pct": price_change,
                    "base_currency": base,
                    "quote_currency": quote,
                    "base_strength": current_strength[base],
                    "quote_strength": current_strength[quote],
                    "strength_diff": strength_diff,
                    "expected_direction": "up" if expected_direction > 0 else "down",
                    "actual_direction": "up" if actual_direction > 0 else "down",
                    "divergence_score": abs(strength_diff) * (1 + abs(price_change) / 100)
                })

        # 3. Find divergence between related pairs
        # Group pairs by common base or quote currency
        base_groups = defaultdict(list)
        quote_groups = defaultdict(list)

        for pair in pairs:
            if pair in pair_currencies:
                base, quote = pair_currencies[pair]
                base_groups[base].append(pair)
                quote_groups[quote].append(pair)

        # Check for divergence within each group
        for currency, related_pairs in {**base_groups, **quote_groups}.items():
            if len(related_pairs) < 2:
                continue

            # Calculate price changes for all related pairs
            pair_changes = {}
            for pair in related_pairs:
                if pair not in price_data or price_data[pair].empty or len(price_data[pair]) < periods:
                    continue

                df = price_data[pair]
                close_col = next((col for col in df.columns if col.lower() in ['close', 'price', 'adj close']), None)
                if not close_col:
                    continue

                price_change = (df[close_col].iloc[-1] / df[close_col].iloc[-periods] - 1) * 100
                pair_changes[pair] = price_change

            # Need at least 2 pairs with valid data
            if len(pair_changes) < 2:
                continue

            # Find pairs with divergent price action
            for pair1, change1 in pair_changes.items():
                for pair2, change2 in pair_changes.items():
                    if pair1 >= pair2:  # Avoid duplicates and self-comparison
                        continue

                    # Check if price changes are in opposite directions
                    if change1 * change2 < 0 and abs(change1 - change2) >= threshold * 2:
                        results["cross_pair_divergence"].append({
                            "common_currency": currency,
                            "pair1": pair1,
                            "pair2": pair2,
                            "pair1_change": change1,
                            "pair2_change": change2,
                            "divergence": abs(change1 - change2),
                            "divergence_score": abs(change1 - change2) / (abs(change1) + abs(change2) + 0.001)
                        })

        # 4. Analyze divergence within currency baskets
        # Define common currency baskets
        baskets = {
            "commodity_currencies": ["AUD", "CAD", "NZD"],
            "european_currencies": ["EUR", "GBP", "CHF"],
            "asian_currencies": ["JPY", "SGD", "HKD"],
            "safe_havens": ["USD", "JPY", "CHF"]
        }

        basket_analysis = self.analyze_currency_baskets(baskets, price_data)

        for basket_name, analysis in basket_analysis.items():
            # Check for significant internal divergence
            if analysis.get("strength_divergence", 0) >= threshold:
                results["basket_divergence"].append({
                    "basket": basket_name,
                    "currencies": analysis.get("valid_currencies", []),
                    "strongest": analysis.get("strongest_currency"),
                    "weakest": analysis.get("weakest_currency"),
                    "divergence": analysis.get("strength_divergence"),
                    "average_strength": analysis.get("average_strength", 0)
                })

        # Sort all results by divergence score or magnitude
        if results["pair_divergence"]:
            results["pair_divergence"] = sorted(results["pair_divergence"],
                                              key=lambda x: x.get("divergence_score", 0),
                                              reverse=True)

        if results["cross_pair_divergence"]:
            results["cross_pair_divergence"] = sorted(results["cross_pair_divergence"],
                                                   key=lambda x: x.get("divergence_score", 0),
                                                   reverse=True)

        if results["basket_divergence"]:
            results["basket_divergence"] = sorted(results["basket_divergence"],
                                               key=lambda x: x.get("divergence", 0),
                                               reverse=True)

        return results

    def analyze_currency_baskets(
        self,
        baskets: Dict[str, List[str]],
        price_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze the strength of currency baskets (e.g., commodity currencies, safe havens).

        Args:
            baskets: Dictionary mapping basket names to lists of currencies
            price_data: Dictionary mapping currency pairs to price DataFrames

        Returns:
            Dictionary with basket analysis results
        """
        # Calculate individual currency strength
        currency_strength = self.calculate_currency_strength(price_data)

        basket_results = {}

        for basket_name, currencies in baskets.items():
            # Filter to currencies that we have strength data for
            valid_currencies = [c for c in currencies if c in currency_strength]

            if not valid_currencies:
                basket_results[basket_name] = {
                    "average_strength": 0.0,
                    "currencies": currencies,
                    "valid_currencies": [],
                    "individual_strengths": {}
                }
                continue

            # Calculate average strength for the basket
            basket_strength = sum(currency_strength[c] for c in valid_currencies) / len(valid_currencies)

            # Store individual strengths
            individual_strengths = {c: currency_strength[c] for c in valid_currencies}

            # Find strongest and weakest in the basket
            strongest = max(valid_currencies, key=lambda c: currency_strength[c])
            weakest = min(valid_currencies, key=lambda c: currency_strength[c])

            basket_results[basket_name] = {
                "average_strength": basket_strength,
                "currencies": currencies,
                "valid_currencies": valid_currencies,
                "individual_strengths": individual_strengths,
                "strongest_currency": strongest,
                "weakest_currency": weakest,
                "strength_divergence": currency_strength[strongest] - currency_strength[weakest]
            }

        return basket_results

    def find_pair_opportunities(
        self,
        price_data: Dict[str, pd.DataFrame],
        min_strength_difference: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find trading opportunities based on currency strength divergence.

        Args:
            price_data: Dictionary mapping currency pairs to price DataFrames
            min_strength_difference: Minimum strength difference to consider

        Returns:
            List of potential trading opportunities
        """
        # Calculate currency strength
        currency_strength = self.calculate_currency_strength(price_data)

        opportunities = []

        # Find all possible pairs from our currencies
        for base in currency_strength:
            for quote in currency_strength:
                if base == quote:
                    continue

                pair = f"{base}/{quote}"

                # Calculate strength difference
                strength_diff = currency_strength[base] - currency_strength[quote]

                # Check if difference exceeds threshold
                if abs(strength_diff) >= min_strength_difference:
                    # Determine if the pair exists in our data or needs to be inverted
                    if pair in price_data:
                        existing_pair = pair
                        invert_signal = False
                    else:
                        inverted_pair = f"{quote}/{base}"
                        if inverted_pair in price_data:
                            existing_pair = inverted_pair
                            invert_signal = True
                        else:
                            # Skip if we don't have data for this pair
                            continue

                    # Determine trade direction
                    if strength_diff > 0:  # Base is stronger than quote
                        direction = "buy" if not invert_signal else "sell"
                    else:  # Quote is stronger than base
                        direction = "sell" if not invert_signal else "buy"

                    opportunities.append({
                        "pair": pair,
                        "existing_pair": existing_pair,
                        "base_currency": base,
                        "quote_currency": quote,
                        "base_strength": currency_strength[base],
                        "quote_strength": currency_strength[quote],
                        "strength_difference": abs(strength_diff),
                        "direction": direction,
                        "invert_signal": invert_signal
                    })

        # Sort by strength difference (descending)
        return sorted(opportunities, key=lambda x: x["strength_difference"], reverse=True)
