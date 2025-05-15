"""
Market Regime module for Market Analysis Service.

This module provides algorithms for detecting market regimes in market data.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
from market_analysis_service.models.market_analysis_models import MarketRegimeType

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Class for detecting market regimes in market data.
    """

    def __init__(self):
        """
        Initialize the Market Regime Detector.
        """
        self.available_regimes = self._get_available_regimes()

    def _get_available_regimes(self) -> List[Dict[str, Any]]:
        """
        Get available market regimes for detection.

        Returns:
            List of available regimes
        """
        regimes = []

        for regime_type in MarketRegimeType:
            regime_info = {
                "id": regime_type.value,
                "name": regime_type.name,
                "description": self._get_regime_description(regime_type)
            }

            regimes.append(regime_info)

        return regimes

    def _get_regime_description(self, regime_type: MarketRegimeType) -> str:
        """
        Get description for a regime type.

        Args:
            regime_type: Regime type

        Returns:
            Regime description
        """
        descriptions = {
            MarketRegimeType.TRENDING_UP: "Market is in an uptrend",
            MarketRegimeType.TRENDING_DOWN: "Market is in a downtrend",
            MarketRegimeType.RANGING: "Market is moving sideways in a range",
            MarketRegimeType.VOLATILE: "Market is experiencing high volatility",
            MarketRegimeType.BREAKOUT: "Market is breaking out of a range",
            MarketRegimeType.REVERSAL: "Market is reversing its trend",
            MarketRegimeType.CONSOLIDATION: "Market is consolidating after a trend",
            MarketRegimeType.CUSTOM: "Custom regime defined by user parameters"
        }

        return descriptions.get(regime_type, "Unknown regime")

    def detect_market_regime(
        self,
        data: pd.DataFrame,
        window_size: int = 20,
        additional_parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect market regimes in market data.

        Args:
            data: Market data
            window_size: Window size for regime detection
            additional_parameters: Additional parameters for detection

        Returns:
            List of detected regimes
        """
        if additional_parameters is None:
            additional_parameters = {}

        # Ensure we have enough data
        if len(data) < window_size:
            return []

        # Detect regimes
        regimes = []

        # Detect trend
        trend_regimes = self._detect_trend(data, window_size, additional_parameters)
        regimes.extend(trend_regimes)

        # Detect volatility
        volatility_regimes = self._detect_volatility(data, window_size, additional_parameters)
        regimes.extend(volatility_regimes)

        # Detect range
        range_regimes = self._detect_range(data, window_size, additional_parameters)
        regimes.extend(range_regimes)

        # Detect breakout
        breakout_regimes = self._detect_breakout(data, window_size, additional_parameters)
        regimes.extend(breakout_regimes)

        # Detect reversal
        reversal_regimes = self._detect_reversal(data, window_size, additional_parameters)
        regimes.extend(reversal_regimes)

        # Detect consolidation
        consolidation_regimes = self._detect_consolidation(data, window_size, additional_parameters)
        regimes.extend(consolidation_regimes)

        # Sort regimes by start date
        regimes.sort(key=lambda x: x["start_index"])

        return regimes

    def _detect_trend(
        self,
        data: pd.DataFrame,
        window_size: int,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect trending regimes in market data.

        Args:
            data: Market data
            window_size: Window size for detection
            parameters: Additional parameters

        Returns:
            List of detected regimes
        """
        regimes = []

        # Get parameters
        min_trend_length = parameters.get("min_trend_length", 10)
        trend_threshold = parameters.get("trend_threshold", 0.6)

        # Calculate moving averages
        short_ma = data["close"].rolling(window=window_size).mean()
        long_ma = data["close"].rolling(window=window_size * 2).mean()

        # Calculate trend direction
        trend_direction = np.zeros(len(data))

        for i in range(window_size * 2, len(data)):
            if short_ma.iloc[i] > long_ma.iloc[i]:
                trend_direction[i] = 1  # Uptrend
            elif short_ma.iloc[i] < long_ma.iloc[i]:
                trend_direction[i] = -1  # Downtrend

        # Find trend segments
        current_trend = 0
        trend_start = 0

        for i in range(window_size * 2, len(data)):
            if trend_direction[i] != current_trend:
                # End of previous trend
                if current_trend != 0 and i - trend_start >= min_trend_length:
                    regime_type = MarketRegimeType.TRENDING_UP if current_trend == 1 else MarketRegimeType.TRENDING_DOWN

                    # Calculate confidence
                    if current_trend == 1:
                        # Uptrend confidence based on price increase
                        price_change = (data["close"].iloc[i - 1] - data["close"].iloc[trend_start]) / data["close"].iloc[trend_start]
                        confidence = min(1.0, max(0.0, price_change / trend_threshold))
                    else:
                        # Downtrend confidence based on price decrease
                        price_change = (data["close"].iloc[trend_start] - data["close"].iloc[i - 1]) / data["close"].iloc[trend_start]
                        confidence = min(1.0, max(0.0, price_change / trend_threshold))

                    regimes.append({
                        "regime_type": regime_type.value,
                        "start_index": int(trend_start),
                        "end_index": int(i - 1),
                        "start_date": data.index[trend_start].isoformat() if hasattr(data.index[trend_start], 'isoformat') else str(data.index[trend_start]),
                        "end_date": data.index[i - 1].isoformat() if hasattr(data.index[i - 1], 'isoformat') else str(data.index[i - 1]),
                        "confidence": float(confidence),
                        "metadata": {
                            "price_change": float(price_change)
                        }
                    })

                # Start of new trend
                current_trend = trend_direction[i]
                trend_start = i

        # Add the last trend if it's still active
        if current_trend != 0 and len(data) - trend_start >= min_trend_length:
            regime_type = MarketRegimeType.TRENDING_UP if current_trend == 1 else MarketRegimeType.TRENDING_DOWN

            # Calculate confidence
            if current_trend == 1:
                # Uptrend confidence based on price increase
                price_change = (data["close"].iloc[-1] - data["close"].iloc[trend_start]) / data["close"].iloc[trend_start]
                confidence = min(1.0, max(0.0, price_change / trend_threshold))
            else:
                # Downtrend confidence based on price decrease
                price_change = (data["close"].iloc[trend_start] - data["close"].iloc[-1]) / data["close"].iloc[trend_start]
                confidence = min(1.0, max(0.0, price_change / trend_threshold))

            regimes.append({
                "regime_type": regime_type.value,
                "start_index": int(trend_start),
                "end_index": int(len(data) - 1),
                "start_date": data.index[trend_start].isoformat() if hasattr(data.index[trend_start], 'isoformat') else str(data.index[trend_start]),
                "end_date": data.index[-1].isoformat() if hasattr(data.index[-1], 'isoformat') else str(data.index[-1]),
                "confidence": float(confidence),
                "metadata": {
                    "price_change": float(price_change)
                }
            })

        return regimes

    def _detect_volatility(
        self,
        data: pd.DataFrame,
        window_size: int,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect volatile regimes in market data.

        Args:
            data: Market data
            window_size: Window size for detection
            parameters: Additional parameters

        Returns:
            List of detected regimes
        """
        regimes = []

        # Get parameters
        volatility_threshold = parameters.get("volatility_threshold", 0.02)
        min_volatile_length = parameters.get("min_volatile_length", 5)

        # Calculate volatility (standard deviation of returns)
        returns = data["close"].pct_change()
        volatility = returns.rolling(window=window_size).std()

        # Find volatile segments
        is_volatile = volatility > volatility_threshold

        volatile_start = None

        for i in range(window_size, len(data)):
            if is_volatile.iloc[i] and volatile_start is None:
                # Start of volatile segment
                volatile_start = i
            elif not is_volatile.iloc[i] and volatile_start is not None:
                # End of volatile segment
                if i - volatile_start >= min_volatile_length:
                    # Calculate confidence
                    avg_volatility = volatility.iloc[volatile_start:i].mean()
                    confidence = min(1.0, avg_volatility / volatility_threshold)

                    regimes.append({
                        "regime_type": MarketRegimeType.VOLATILE.value,
                        "start_index": int(volatile_start),
                        "end_index": int(i - 1),
                        "start_date": data.index[volatile_start].isoformat() if hasattr(data.index[volatile_start], 'isoformat') else str(data.index[volatile_start]),
                        "end_date": data.index[i - 1].isoformat() if hasattr(data.index[i - 1], 'isoformat') else str(data.index[i - 1]),
                        "confidence": float(confidence),
                        "metadata": {
                            "avg_volatility": float(avg_volatility)
                        }
                    })

                volatile_start = None

        # Add the last volatile segment if it's still active
        if volatile_start is not None and len(data) - volatile_start >= min_volatile_length:
            # Calculate confidence
            avg_volatility = volatility.iloc[volatile_start:].mean()
            confidence = min(1.0, avg_volatility / volatility_threshold)

            regimes.append({
                "regime_type": MarketRegimeType.VOLATILE.value,
                "start_index": int(volatile_start),
                "end_index": int(len(data) - 1),
                "start_date": data.index[volatile_start].isoformat() if hasattr(data.index[volatile_start], 'isoformat') else str(data.index[volatile_start]),
                "end_date": data.index[-1].isoformat() if hasattr(data.index[-1], 'isoformat') else str(data.index[-1]),
                "confidence": float(confidence),
                "metadata": {
                    "avg_volatility": float(avg_volatility)
                }
            })

        return regimes

    def _detect_range(
        self,
        data: pd.DataFrame,
        window_size: int,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect ranging regimes in market data.

        Args:
            data: Market data
            window_size: Window size for detection
            parameters: Additional parameters

        Returns:
            List of detected regimes
        """
        regimes = []

        # Get parameters
        range_threshold = parameters.get("range_threshold", 0.03)
        min_range_length = parameters.get("min_range_length", 10)

        # Calculate price range
        rolling_high = data["high"].rolling(window=window_size).max()
        rolling_low = data["low"].rolling(window=window_size).min()
        price_range = (rolling_high - rolling_low) / rolling_low

        # Find ranging segments
        is_ranging = price_range < range_threshold

        range_start = None

        for i in range(window_size, len(data)):
            if is_ranging.iloc[i] and range_start is None:
                # Start of ranging segment
                range_start = i
            elif not is_ranging.iloc[i] and range_start is not None:
                # End of ranging segment
                if i - range_start >= min_range_length:
                    # Calculate confidence
                    avg_range = price_range.iloc[range_start:i].mean()
                    confidence = min(1.0, 1 - avg_range / range_threshold)

                    regimes.append({
                        "regime_type": MarketRegimeType.RANGING.value,
                        "start_index": int(range_start),
                        "end_index": int(i - 1),
                        "start_date": data.index[range_start].isoformat() if hasattr(data.index[range_start], 'isoformat') else str(data.index[range_start]),
                        "end_date": data.index[i - 1].isoformat() if hasattr(data.index[i - 1], 'isoformat') else str(data.index[i - 1]),
                        "confidence": float(confidence),
                        "metadata": {
                            "avg_range": float(avg_range)
                        }
                    })

                range_start = None

        # Add the last ranging segment if it's still active
        if range_start is not None and len(data) - range_start >= min_range_length:
            # Calculate confidence
            avg_range = price_range.iloc[range_start:].mean()
            confidence = min(1.0, 1 - avg_range / range_threshold)

            regimes.append({
                "regime_type": MarketRegimeType.RANGING.value,
                "start_index": int(range_start),
                "end_index": int(len(data) - 1),
                "start_date": data.index[range_start].isoformat() if hasattr(data.index[range_start], 'isoformat') else str(data.index[range_start]),
                "end_date": data.index[-1].isoformat() if hasattr(data.index[-1], 'isoformat') else str(data.index[-1]),
                "confidence": float(confidence),
                "metadata": {
                    "avg_range": float(avg_range)
                }
            })

        return regimes

    def _detect_breakout(
        self,
        data: pd.DataFrame,
        window_size: int,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect breakout regimes in market data.

        Args:
            data: Market data
            window_size: Window size for detection
            parameters: Additional parameters

        Returns:
            List of detected regimes
        """
        regimes = []

        # Get parameters
        breakout_threshold = parameters.get("breakout_threshold", 0.02)
        min_consolidation_length = parameters.get("min_consolidation_length", 10)

        # Calculate price range
        rolling_high = data["high"].rolling(window=window_size).max()
        rolling_low = data["low"].rolling(window=window_size).min()

        # Find breakouts
        for i in range(window_size * 2, len(data)):
            # Check if there was a consolidation period before
            prev_range = (rolling_high.iloc[i - window_size] - rolling_low.iloc[i - window_size]) / rolling_low.iloc[i - window_size]

            if prev_range < breakout_threshold:
                # Check if price broke out of the range
                if data["close"].iloc[i] > rolling_high.iloc[i - window_size]:
                    # Upside breakout
                    breakout_size = (data["close"].iloc[i] - rolling_high.iloc[i - window_size]) / rolling_high.iloc[i - window_size]

                    if breakout_size > breakout_threshold:
                        # Calculate confidence
                        confidence = min(1.0, breakout_size / breakout_threshold)

                        regimes.append({
                            "regime_type": MarketRegimeType.BREAKOUT.value,
                            "start_index": int(i - min_consolidation_length),
                            "end_index": int(i),
                            "start_date": data.index[i - min_consolidation_length].isoformat() if hasattr(data.index[i - min_consolidation_length], 'isoformat') else str(data.index[i - min_consolidation_length]),
                            "end_date": data.index[i].isoformat() if hasattr(data.index[i], 'isoformat') else str(data.index[i]),
                            "confidence": float(confidence),
                            "metadata": {
                                "breakout_direction": "up",
                                "breakout_size": float(breakout_size)
                            }
                        })

                elif data["close"].iloc[i] < rolling_low.iloc[i - window_size]:
                    # Downside breakout
                    breakout_size = (rolling_low.iloc[i - window_size] - data["close"].iloc[i]) / rolling_low.iloc[i - window_size]

                    if breakout_size > breakout_threshold:
                        # Calculate confidence
                        confidence = min(1.0, breakout_size / breakout_threshold)

                        regimes.append({
                            "regime_type": MarketRegimeType.BREAKOUT.value,
                            "start_index": int(i - min_consolidation_length),
                            "end_index": int(i),
                            "start_date": data.index[i - min_consolidation_length].isoformat() if hasattr(data.index[i - min_consolidation_length], 'isoformat') else str(data.index[i - min_consolidation_length]),
                            "end_date": data.index[i].isoformat() if hasattr(data.index[i], 'isoformat') else str(data.index[i]),
                            "confidence": float(confidence),
                            "metadata": {
                                "breakout_direction": "down",
                                "breakout_size": float(breakout_size)
                            }
                        })

        return regimes

    def _detect_reversal(
        self,
        data: pd.DataFrame,
        window_size: int,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect reversal regimes in market data.

        Args:
            data: Market data
            window_size: Window size for detection
            parameters: Additional parameters

        Returns:
            List of detected regimes
        """
        regimes = []

        # Get parameters
        reversal_threshold = parameters.get("reversal_threshold", 0.05)

        # Calculate moving averages
        short_ma = data["close"].rolling(window=window_size).mean()
        long_ma = data["close"].rolling(window=window_size * 2).mean()

        # Calculate trend direction
        trend_direction = np.zeros(len(data))

        for i in range(window_size * 2, len(data)):
            if short_ma.iloc[i] > long_ma.iloc[i]:
                trend_direction[i] = 1  # Uptrend
            elif short_ma.iloc[i] < long_ma.iloc[i]:
                trend_direction[i] = -1  # Downtrend

        # Find trend reversals
        for i in range(window_size * 3, len(data)):
            # Check if there was a trend change
            if trend_direction[i] != 0 and trend_direction[i] != trend_direction[i - window_size]:
                # Calculate price change
                if trend_direction[i] == 1:
                    # Reversal from downtrend to uptrend
                    price_change = (data["close"].iloc[i] - data["close"].iloc[i - window_size]) / data["close"].iloc[i - window_size]

                    if price_change > reversal_threshold:
                        # Calculate confidence
                        confidence = min(1.0, price_change / reversal_threshold)

                        regimes.append({
                            "regime_type": MarketRegimeType.REVERSAL.value,
                            "start_index": int(i - window_size),
                            "end_index": int(i),
                            "start_date": data.index[i - window_size].isoformat() if hasattr(data.index[i - window_size], 'isoformat') else str(data.index[i - window_size]),
                            "end_date": data.index[i].isoformat() if hasattr(data.index[i], 'isoformat') else str(data.index[i]),
                            "confidence": float(confidence),
                            "metadata": {
                                "reversal_direction": "up",
                                "price_change": float(price_change)
                            }
                        })

                elif trend_direction[i] == -1:
                    # Reversal from uptrend to downtrend
                    price_change = (data["close"].iloc[i - window_size] - data["close"].iloc[i]) / data["close"].iloc[i - window_size]

                    if price_change > reversal_threshold:
                        # Calculate confidence
                        confidence = min(1.0, price_change / reversal_threshold)

                        regimes.append({
                            "regime_type": MarketRegimeType.REVERSAL.value,
                            "start_index": int(i - window_size),
                            "end_index": int(i),
                            "start_date": data.index[i - window_size].isoformat() if hasattr(data.index[i - window_size], 'isoformat') else str(data.index[i - window_size]),
                            "end_date": data.index[i].isoformat() if hasattr(data.index[i], 'isoformat') else str(data.index[i]),
                            "confidence": float(confidence),
                            "metadata": {
                                "reversal_direction": "down",
                                "price_change": float(price_change)
                            }
                        })

        return regimes

    def _detect_consolidation(
        self,
        data: pd.DataFrame,
        window_size: int,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect consolidation regimes in market data.

        Args:
            data: Market data
            window_size: Window size for detection
            parameters: Additional parameters

        Returns:
            List of detected regimes
        """
        regimes = []

        # Get parameters
        consolidation_threshold = parameters.get("consolidation_threshold", 0.02)
        min_consolidation_length = parameters.get("min_consolidation_length", 10)

        # Calculate price range
        rolling_high = data["high"].rolling(window=window_size).max()
        rolling_low = data["low"].rolling(window=window_size).min()
        price_range = (rolling_high - rolling_low) / rolling_low

        # Calculate volatility
        returns = data["close"].pct_change()
        volatility = returns.rolling(window=window_size).std()

        # Find consolidation segments
        is_consolidating = (price_range < consolidation_threshold) & (volatility < consolidation_threshold / 2)

        consolidation_start = None

        for i in range(window_size, len(data)):
            if is_consolidating.iloc[i] and consolidation_start is None:
                # Start of consolidation segment
                consolidation_start = i
            elif not is_consolidating.iloc[i] and consolidation_start is not None:
                # End of consolidation segment
                if i - consolidation_start >= min_consolidation_length:
                    # Calculate confidence
                    avg_range = price_range.iloc[consolidation_start:i].mean()
                    confidence = min(1.0, 1 - avg_range / consolidation_threshold)

                    regimes.append({
                        "regime_type": MarketRegimeType.CONSOLIDATION.value,
                        "start_index": int(consolidation_start),
                        "end_index": int(i - 1),
                        "start_date": data.index[consolidation_start].isoformat() if hasattr(data.index[consolidation_start], 'isoformat') else str(data.index[consolidation_start]),
                        "end_date": data.index[i - 1].isoformat() if hasattr(data.index[i - 1], 'isoformat') else str(data.index[i - 1]),
                        "confidence": float(confidence),
                        "metadata": {
                            "avg_range": float(avg_range)
                        }
                    })

                consolidation_start = None

        # Add the last consolidation segment if it's still active
        if consolidation_start is not None and len(data) - consolidation_start >= min_consolidation_length:
            # Calculate confidence
            avg_range = price_range.iloc[consolidation_start:].mean()
            confidence = min(1.0, 1 - avg_range / consolidation_threshold)

            regimes.append({
                "regime_type": MarketRegimeType.CONSOLIDATION.value,
                "start_index": int(consolidation_start),
                "end_index": int(len(data) - 1),
                "start_date": data.index[consolidation_start].isoformat() if hasattr(data.index[consolidation_start], 'isoformat') else str(data.index[consolidation_start]),
                "end_date": data.index[-1].isoformat() if hasattr(data.index[-1], 'isoformat') else str(data.index[-1]),
                "confidence": float(confidence),
                "metadata": {
                    "avg_range": float(avg_range)
                }
            })

        return regimes