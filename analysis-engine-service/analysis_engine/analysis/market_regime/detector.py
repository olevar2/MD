"""
Market Regime Detector

This module provides functionality to detect market regimes based on price data.
It calculates technical indicators and extracts features for regime classification.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

from analysis_engine.caching.cache_service import cache_result
from analysis_engine.repositories.price_repository import PriceRepository

from .models import MarketRegimeResult, MarketRegimeType, TrendState, VolatilityState

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Detector for market regimes based on price action analysis.

    This class is responsible for calculating technical indicators and
    extracting features from price data for regime classification.
    """

    def __init__(self, price_repository: PriceRepository):
        """
        Initialize the market regime detector.

        Args:
            price_repository: Repository for fetching price data
        """
        self.price_repository = price_repository
        logger.info("MarketRegimeDetector initialized")

    @cache_result(ttl=900)  # Cache for 15 minutes
    async def get_price_data(
        self,
        instrument: str,
        timeframe: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> List[Any]:
        """
        Get price data for analysis.

        Args:
            instrument: The instrument to analyze
            timeframe: The timeframe to analyze
            from_date: Start date for analysis
            to_date: End date for analysis

        Returns:
            List[Any]: Price data
        """
        return self.price_repository.get_prices(
            instrument=instrument,
            timeframe=timeframe,
            from_date=from_date,
            to_date=to_date
        )

    @cache_result(ttl=900)  # Cache for 15 minutes
    def calculate_atr(self, prices: List[Any], period: int = 14) -> float:
        """
        Calculate Average True Range.

        Args:
            prices: List of price objects
            period: Period for calculation

        Returns:
            float: ATR value
        """
        if len(prices) < period + 1:
            return 0.0

        true_ranges = []
        for i in range(1, len(prices)):
            high = prices[i].high
            low = prices[i].low
            prev_close = prices[i-1].close

            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)

            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)

        return np.mean(true_ranges[-period:])

    @cache_result(ttl=900)  # Cache for 15 minutes
    def calculate_adx(self, prices: List[Any], period: int = 14) -> float:
        """
        Calculate Average Directional Index.

        Args:
            prices: List of price objects
            period: Period for calculation

        Returns:
            float: ADX value
        """
        if len(prices) < period * 2:
            return 0.0

        # Calculate +DI and -DI
        plus_dm = []
        minus_dm = []
        tr = []

        for i in range(1, len(prices)):
            high = prices[i].high
            low = prices[i].low
            prev_high = prices[i-1].high
            prev_low = prices[i-1].low
            prev_close = prices[i-1].close

            # True Range
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            tr.append(max(tr1, tr2, tr3))

            # Directional Movement
            up_move = high - prev_high
            down_move = prev_low - low

            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)

            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)

        # Smooth with EMA
        tr_ema = self._calculate_ema(tr, period)
        plus_dm_ema = self._calculate_ema(plus_dm, period)
        minus_dm_ema = self._calculate_ema(minus_dm, period)

        # Calculate DI
        plus_di = 100 * plus_dm_ema / tr_ema if tr_ema > 0 else 0
        minus_di = 100 * minus_dm_ema / tr_ema if tr_ema > 0 else 0

        # Calculate DX
        if plus_di + minus_di > 0:
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        else:
            dx = 0

        # Calculate ADX (smoothed DX)
        adx_values = [dx]
        for i in range(1, period):
            if i >= len(prices) - period:
                break
            adx_values.append((adx_values[-1] * (period - 1) + dx) / period)

        return adx_values[-1] if adx_values else 0

    @cache_result(ttl=900)  # Cache for 15 minutes
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """
        Calculate Relative Strength Index.

        Args:
            prices: Array of price values
            period: Period for calculation

        Returns:
            float: RSI value
        """
        if len(prices) <= period:
            return 50.0  # Default neutral value

        # Calculate price changes
        deltas = np.diff(prices)

        # Separate gains and losses
        gains = deltas.copy()
        losses = deltas.copy()

        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        # Calculate initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        if avg_loss == 0:
            return 100.0

        # Calculate RS and RSI
        rs = avg_gain / avg_loss if avg_loss > 0 else 1000
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @cache_result(ttl=900)  # Cache for 15 minutes
    def _calculate_ema(self, values: List[float], period: int) -> float:
        """
        Calculate Exponential Moving Average.

        Args:
            values: List of values
            period: Period for calculation

        Returns:
            float: EMA value
        """
        if not values or period <= 0 or len(values) < period:
            return 0.0

        ema = sum(values[:period]) / period
        multiplier = 2 / (period + 1)

        for i in range(period, len(values)):
            ema = (values[i] - ema) * multiplier + ema

        return ema

    @cache_result(ttl=900)  # Cache for 15 minutes
    def extract_features(self, prices: List[Any]) -> Dict[str, float]:
        """
        Extract features from price data for regime classification.

        Args:
            prices: List of price objects

        Returns:
            Dict[str, float]: Extracted features
        """
        if not prices or len(prices) < 50:
            return {}

        # Extract price series
        close_prices = np.array([p.close for p in prices])

        # Calculate indicators
        atr = self.calculate_atr(prices)
        adx = self.calculate_adx(prices)
        rsi = self.calculate_rsi(close_prices)

        # Calculate moving averages
        sma20 = np.mean(close_prices[-20:])
        sma50 = np.mean(close_prices[-50:]) if len(close_prices) >= 50 else sma20

        # Calculate volatility metrics
        avg_range = np.mean([p.high - p.low for p in prices])
        avg_price = np.mean(close_prices)
        volatility_ratio = (avg_range / avg_price) * 100  # As percentage of price

        # Calculate trend metrics
        price_change = (close_prices[-1] / close_prices[0] - 1) * 100  # Percentage change
        ma_diff = (sma20 / sma50 - 1) * 100  # Percentage difference between MAs

        return {
            "atr": atr,
            "adx": adx,
            "rsi": rsi,
            "sma20": sma20,
            "sma50": sma50,
            "volatility_ratio": volatility_ratio,
            "price_change": price_change,
            "ma_diff": ma_diff,
            "close": close_prices[-1]
        }