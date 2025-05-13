"""
Optimized Confluence Detector

This module provides an optimized implementation of confluence detection for forex trading.
It detects confluence signals across related currency pairs with improved performance
and reduced memory usage.

Features:
- Vectorized operations for faster calculations
- Adaptive caching for improved performance
- Parallel processing for multi-pair analysis
- Memory optimization for reduced footprint
- Early termination for faster results
"""
import pandas as pd
import numpy as np
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import asyncio
from common_lib.caching import AdaptiveCacheManager, cached, get_cache_manager
from common_lib.parallel import ParallelProcessor, get_parallel_processor
from analysis_engine.utils.memory_optimized_dataframe import MemoryOptimizedDataFrame
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class OptimizedConfluenceDetector:
    """
    Optimized implementation of confluence detection for forex trading.

    Features:
    - Vectorized operations for faster calculations
    - Adaptive caching for improved performance
    - Parallel processing for multi-pair analysis
    - Memory optimization for reduced footprint
    - Early termination for faster results
    """

    def __init__(self, correlation_service=None, currency_strength_analyzer
        =None, correlation_threshold: float=0.7, lookback_periods: int=20,
        cache_ttl_minutes: int=60, max_workers: int=4):
        """
        Initialize the optimized confluence detector.

        Args:
            correlation_service: Service for retrieving correlations between pairs
            currency_strength_analyzer: Analyzer for currency strength
            correlation_threshold: Minimum correlation to consider pairs related
            lookback_periods: Number of periods to look back for analysis
            cache_ttl_minutes: Cache time-to-live in minutes
            max_workers: Maximum number of parallel workers
        """
        self.correlation_service = correlation_service
        self.currency_strength_analyzer = currency_strength_analyzer
        self.correlation_threshold = correlation_threshold
        self.lookback_periods = lookback_periods
        self.max_workers = max_workers
        self.cache_manager = AdaptiveCacheManager(default_ttl_seconds=
            cache_ttl_minutes * 60, max_size=1000, cleanup_interval_seconds=300
            )
        self.parallel_processor = OptimizedParallelProcessor(min_workers=2,
            max_workers=max_workers)
        logger.debug(
            f'OptimizedConfluenceDetector initialized with correlation_threshold={correlation_threshold}, lookback_periods={lookback_periods}, cache_ttl_minutes={cache_ttl_minutes}, max_workers={max_workers}'
            )

    @async_with_exception_handling
    async def find_related_pairs(self, symbol: str) ->Dict[str, float]:
        """
        Find pairs related to the given symbol based on correlation.

        Args:
            symbol: The primary currency pair

        Returns:
            Dictionary mapping related pairs to their correlation values
        """
        cache_key = f'related_pairs_{symbol}'
        cache_hit, cached_result = self.cache_manager.get(cache_key)
        if cache_hit:
            logger.debug(f'Cache hit for related pairs: {symbol}')
            return cached_result
        if self.correlation_service:
            try:
                all_correlations = (await self.correlation_service.
                    get_all_correlations())
                symbol_correlations = all_correlations.get(symbol, {})
                related_pairs = {pair: corr for pair, corr in
                    symbol_correlations.items() if abs(corr) >= self.
                    correlation_threshold}
                self.cache_manager.set(cache_key, related_pairs)
                return related_pairs
            except Exception as e:
                logger.error(f'Error finding related pairs for {symbol}: {e}',
                    exc_info=True)
                return {}
        else:
            logger.warning(
                f'No correlation service available to find related pairs for {symbol}'
                )
            return {}

    @with_exception_handling
    def detect_confluence_optimized(self, symbol: str, price_data: Dict[str,
        pd.DataFrame], signal_type: str, signal_direction: str,
        related_pairs: Optional[Dict[str, float]]=None,
        use_currency_strength: bool=True, min_confirmation_strength: float=0.3
        ) ->Dict[str, Any]:
        """
        Optimized confluence detection with improved algorithm and memory usage.

        Args:
            symbol: The primary currency pair
            price_data: Dictionary mapping currency pairs to price DataFrames
            signal_type: Type of signal to look for
            signal_direction: Direction of the signal
            related_pairs: Optional dictionary of related pairs
            use_currency_strength: Whether to incorporate currency strength
            min_confirmation_strength: Minimum signal strength to consider

        Returns:
            Dictionary with confluence analysis results
        """
        start_time = time.time()
        if symbol not in price_data or price_data[symbol].empty:
            return {'error': f'No price data available for {symbol}'}
        cache_key = (
            f'confluence_{symbol}_{signal_type}_{signal_direction}_{hash(frozenset(related_pairs.items()) if related_pairs else None)}'
            )
        cache_hit, cached_result = self.cache_manager.get(cache_key)
        if cache_hit:
            logger.debug(f'Cache hit for confluence detection: {cache_key}')
            return cached_result
        if related_pairs is None:
            try:
                if asyncio.get_event_loop().is_running():
                    related_pairs = asyncio.run(self.find_related_pairs(symbol)
                        )
                else:
                    loop = asyncio.new_event_loop()
                    related_pairs = loop.run_until_complete(self.
                        find_related_pairs(symbol))
                    loop.close()
            except Exception as e:
                logger.error(f'Error finding related pairs: {e}', exc_info=True
                    )
                related_pairs = {}
        available_related = {pair: corr for pair, corr in related_pairs.
            items() if pair in price_data and not price_data[pair].empty}
        if not available_related:
            result = {'symbol': symbol, 'signal_type': signal_type,
                'signal_direction': signal_direction, 'confirmation_count':
                0, 'contradiction_count': 0, 'confluence_score': 0.0,
                'message': 'No related pairs with available data'}
            self.cache_manager.set(cache_key, result)
            return result
        currency_strength = {}
        if use_currency_strength and self.currency_strength_analyzer:
            try:
                currency_strength = (self.currency_strength_analyzer.
                    calculate_currency_strength(price_data))
            except Exception as e:
                logger.warning(f'Error calculating currency strength: {e}')
        tasks = []
        for related_pair, correlation in available_related.items():
            priority = -abs(correlation)
            tasks.append((priority, self._analyze_pair_signal, (
                related_pair, correlation, price_data[related_pair],
                signal_type, signal_direction, currency_strength)))
        results = self.parallel_processor.process(tasks, timeout=5.0)
        confirmations = []
        contradictions = []
        neutrals = []
        for result in results.values():
            if result is None:
                continue
            signal_result, details = result
            if signal_result == 'confirm':
                confirmations.append(details)
            elif signal_result == 'contradict':
                contradictions.append(details)
            else:
                neutrals.append(details)
        confirmation_count = len(confirmations)
        contradiction_count = len(contradictions)
        total_pairs = confirmation_count + contradiction_count
        weighted_confirmations = sum(details.get('correlation', 0) *
            details.get('signal_strength', 0) for details in confirmations)
        weighted_contradictions = sum(details.get('correlation', 0) *
            details.get('signal_strength', 0) for details in contradictions)
        confluence_score = 0.0
        if total_pairs > 0:
            raw_score = (weighted_confirmations - weighted_contradictions
                ) / total_pairs
            confluence_score = max(0.0, min(1.0, (raw_score + 1) / 2))
            if confirmation_count > 0:
                confirmation_ratio = confirmation_count / total_pairs
                if confirmation_ratio > 0.8:
                    confluence_score = min(1.0, confluence_score * 1.2)
        result = {'symbol': symbol, 'signal_type': signal_type,
            'signal_direction': signal_direction, 'confirmations':
            confirmations, 'contradictions': contradictions, 'neutrals':
            neutrals, 'confirmation_count': confirmation_count,
            'contradiction_count': contradiction_count,
            'weighted_confirmations': weighted_confirmations,
            'weighted_contradictions': weighted_contradictions,
            'confluence_score': confluence_score, 'execution_time': time.
            time() - start_time}
        self.cache_manager.set(cache_key, result)
        return result

    def _analyze_pair_signal(self, related_pair: str, correlation: float,
        price_data: pd.DataFrame, signal_type: str, signal_direction: str,
        currency_strength: Dict[str, float]) ->Tuple[str, Dict[str, Any]]:
        """
        Analyze a single related pair for signal confirmation or contradiction.

        Args:
            related_pair: The related currency pair
            correlation: Correlation with the primary pair
            price_data: Price DataFrame for the related pair
            signal_type: Type of signal to look for
            signal_direction: Direction of the signal
            currency_strength: Dictionary of currency strength values

        Returns:
            Tuple of (result_type, details) where result_type is "confirm", "contradict", or "neutral"
        """
        optimized_data = MemoryOptimizedDataFrame(price_data)
        signal_strength = 0.0
        if signal_type == 'trend':
            signal_strength = self._calculate_trend_strength(optimized_data,
                signal_direction)
        elif signal_type == 'reversal':
            signal_strength = self._calculate_reversal_strength(optimized_data,
                signal_direction)
        elif signal_type == 'breakout':
            signal_strength = self._calculate_breakout_strength(optimized_data,
                signal_direction)
        else:
            return 'neutral', {'pair': related_pair, 'correlation':
                correlation, 'signal_strength': 0.0, 'message':
                f'Unknown signal type: {signal_type}'}
        if currency_strength and len(related_pair) >= 6:
            base_currency = related_pair[:3]
            quote_currency = related_pair[3:6]
            if (base_currency in currency_strength and quote_currency in
                currency_strength):
                base_strength = currency_strength[base_currency]
                quote_strength = currency_strength[quote_currency]
                if signal_direction == 'bullish':
                    currency_factor = (base_strength - quote_strength) / 2
                else:
                    currency_factor = (quote_strength - base_strength) / 2
                signal_strength = signal_strength * 0.7 + currency_factor * 0.3
        expected_direction = signal_direction
        if correlation < 0:
            expected_direction = ('bearish' if signal_direction ==
                'bullish' else 'bullish')
        actual_direction = 'bullish' if signal_strength > 0 else 'bearish'
        details = {'pair': related_pair, 'correlation': correlation,
            'signal_strength': abs(signal_strength), 'expected_direction':
            expected_direction, 'actual_direction': actual_direction}
        if abs(signal_strength) < 0.2:
            return 'neutral', details
        elif expected_direction == actual_direction:
            return 'confirm', details
        else:
            return 'contradict', details

    def _calculate_trend_strength(self, price_data:
        MemoryOptimizedDataFrame, direction: str) ->float:
        """
        Calculate trend strength with vectorized operations.

        Args:
            price_data: Price DataFrame
            direction: Signal direction ("bullish" or "bearish")

        Returns:
            Trend strength (-1.0 to 1.0)
        """
        close_col = next((col for col in price_data.columns if col.lower() in
            ['close', 'price', 'adj close']), None)
        if not close_col or len(price_data) < self.lookback_periods:
            return 0.0
        close_prices = price_data[close_col].values
        if len(close_prices) == 0 or np.isnan(close_prices[-1]):
            return 0.0
        ma_short = np.mean(close_prices[-10:])
        ma_long = np.mean(close_prices[-30:]) if len(close_prices
            ) >= 30 else np.mean(close_prices)
        ma_direction = (ma_short / ma_long - 1) * 5
        momentum_periods = min(14, len(close_prices) - 1)
        momentum = (close_prices[-1] / close_prices[-momentum_periods - 1] - 1
            ) * 10
        if len(close_prices) >= 20:
            x = np.arange(20)
            y = close_prices[-20:]
            slope, _ = np.polyfit(x, y, 1)
            slope_normalized = slope * 200 / np.mean(close_prices[-20:])
        else:
            slope_normalized = 0
        trend_strength = (0.4 * ma_direction + 0.4 * momentum + 0.2 *
            slope_normalized)
        if direction == 'bearish':
            trend_strength = -trend_strength
        return np.clip(trend_strength, -1.0, 1.0)

    def _calculate_reversal_strength(self, price_data:
        MemoryOptimizedDataFrame, direction: str) ->float:
        """
        Calculate reversal signal strength with vectorized operations.

        Args:
            price_data: Price DataFrame
            direction: Signal direction ("bullish" or "bearish")

        Returns:
            Reversal strength (-1.0 to 1.0)
        """
        close_col = next((col for col in price_data.columns if col.lower() in
            ['close', 'price', 'adj close']), None)
        high_col = next((col for col in price_data.columns if col.lower() ==
            'high'), None)
        low_col = next((col for col in price_data.columns if col.lower() ==
            'low'), None)
        if not all([close_col, high_col, low_col]) or len(price_data) < 20:
            return 0.0
        close_prices = price_data[close_col].values
        high_prices = price_data[high_col].values
        low_prices = price_data[low_col].values
        deltas = np.diff(close_prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - 100 / (1 + rs)
        rsi_normalized = (rsi - 50) / 50
        recent_range = np.max(high_prices[-5:]) - np.min(low_prices[-5:])
        previous_range = np.max(high_prices[-10:-5]) - np.min(low_prices[-
            10:-5])
        if previous_range > 0:
            range_ratio = recent_range / previous_range
            exhaustion = 1 - min(1, range_ratio)
        else:
            exhaustion = 0
        price_change = (close_prices[-1] / close_prices[-5] - 1) * 10
        if direction == 'bullish':
            reversal_strength = (-0.5 * rsi_normalized + 0.3 * exhaustion -
                0.2 * price_change)
        else:
            reversal_strength = (0.5 * rsi_normalized + 0.3 * exhaustion + 
                0.2 * price_change)
        return np.clip(reversal_strength, -1.0, 1.0)

    def _calculate_breakout_strength(self, price_data:
        MemoryOptimizedDataFrame, direction: str) ->float:
        """
        Calculate breakout signal strength with vectorized operations.

        Args:
            price_data: Price DataFrame
            direction: Signal direction ("bullish" or "bearish")

        Returns:
            Breakout strength (-1.0 to 1.0)
        """
        close_col = next((col for col in price_data.columns if col.lower() in
            ['close', 'price', 'adj close']), None)
        high_col = next((col for col in price_data.columns if col.lower() ==
            'high'), None)
        low_col = next((col for col in price_data.columns if col.lower() ==
            'low'), None)
        volume_col = next((col for col in price_data.columns if col.lower() ==
            'volume'), None)
        if not close_col or len(price_data) < 20:
            return 0.0
        close_prices = price_data[close_col].values
        if high_col and low_col:
            high_prices = price_data[high_col].values
            low_prices = price_data[low_col].values
            recent_high = np.max(high_prices[-20:-1])
            recent_low = np.min(low_prices[-20:-1])
        else:
            recent_high = np.max(close_prices[-20:-1])
            recent_low = np.min(close_prices[-20:-1])
        current_price = close_prices[-1]
        price_range = recent_high - recent_low
        if price_range > 0:
            relative_position = (current_price - recent_low) / price_range
        else:
            relative_position = 0.5
        momentum_periods = min(5, len(close_prices) - 1)
        momentum = (current_price / close_prices[-momentum_periods - 1] - 1
            ) * 10
        volume_factor = 0
        if volume_col and len(price_data[volume_col]) >= 20:
            volumes = price_data[volume_col].values
            avg_volume = np.mean(volumes[-20:-1])
            current_volume = volumes[-1]
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                volume_factor = min(1, (volume_ratio - 1) / 2)
        if direction == 'bullish':
            breakout_strength = (relative_position - 0.8) * 5
            breakout_strength = (breakout_strength + 0.3 * momentum + 0.2 *
                volume_factor)
        else:
            breakout_strength = (0.2 - relative_position) * 5
            breakout_strength = (breakout_strength - 0.3 * momentum + 0.2 *
                volume_factor)
        return np.clip(breakout_strength, -1.0, 1.0)

    @with_analysis_resilience('analyze_divergence_optimized')
    @with_exception_handling
    def analyze_divergence_optimized(self, symbol: str, price_data: Dict[
        str, pd.DataFrame], related_pairs: Optional[Dict[str, float]]=None
        ) ->Dict[str, Any]:
        """
        Optimized divergence analysis with improved algorithm and memory usage.

        Args:
            symbol: The primary currency pair
            price_data: Dictionary mapping currency pairs to price DataFrames
            related_pairs: Optional dictionary of related pairs

        Returns:
            Dictionary with divergence analysis results
        """
        start_time = time.time()
        if symbol not in price_data or price_data[symbol].empty:
            return {'symbol': symbol, 'divergences_found': 0, 'message':
                'No price data available for primary pair'}
        cache_key = (
            f'divergence_{symbol}_{hash(frozenset(related_pairs.items()) if related_pairs else None)}'
            )
        cache_hit, cached_result = self.cache_manager.get(cache_key)
        if cache_hit:
            logger.debug(f'Cache hit for divergence analysis: {cache_key}')
            return cached_result
        if related_pairs is None:
            try:
                if asyncio.get_event_loop().is_running():
                    related_pairs = asyncio.run(self.find_related_pairs(symbol)
                        )
                else:
                    loop = asyncio.new_event_loop()
                    related_pairs = loop.run_until_complete(self.
                        find_related_pairs(symbol))
                    loop.close()
            except Exception as e:
                logger.error(f'Error finding related pairs: {e}', exc_info=True
                    )
                related_pairs = {}
        available_related = {pair: corr for pair, corr in related_pairs.
            items() if pair in price_data and not price_data[pair].empty}
        if not available_related:
            result = {'symbol': symbol, 'divergences_found': 0, 'message':
                'No related pairs with available data'}
            self.cache_manager.set(cache_key, result)
            return result
        primary_momentum = self._calculate_momentum_cached(price_data[symbol])
        if primary_momentum is None:
            result = {'symbol': symbol, 'divergences_found': 0, 'message':
                'Could not calculate momentum for primary pair'}
            self.cache_manager.set(cache_key, result)
            return result
        tasks = []
        for related_pair, correlation in available_related.items():
            priority = -abs(correlation)
            tasks.append((priority, self._analyze_pair_divergence_optimized,
                (related_pair, correlation, price_data[related_pair],
                primary_momentum)))
        results = self.parallel_processor.process(tasks, timeout=5.0)
        divergences = [result for result in results.values() if result is not
            None]
        divergences.sort(key=lambda x: x.get('divergence_strength', 0),
            reverse=True)
        divergence_score = 0.0
        if divergences:
            weighted_sum = sum(d.get('correlation', 0) * d.get(
                'divergence_strength', 0) for d in divergences)
            total_weight = sum(abs(d.get('correlation', 0)) for d in
                divergences)
            divergence_score = (weighted_sum / total_weight if total_weight >
                0 else 0.0)
            divergence_score = max(0.0, min(1.0, divergence_score))
        result = {'symbol': symbol, 'divergences': divergences,
            'divergences_found': len(divergences), 'divergence_score':
            divergence_score, 'execution_time': time.time() - start_time}
        self.cache_manager.set(cache_key, result)
        return result

    def _calculate_momentum_cached(self, price_data: pd.DataFrame) ->Optional[
        float]:
        """
        Calculate momentum with caching.

        Args:
            price_data: Price DataFrame

        Returns:
            Momentum value or None if calculation fails
        """
        close_col = next((col for col in price_data.columns if col.lower() in
            ['close', 'price', 'adj close']), None)
        if not close_col or len(price_data) < 14:
            return None
        last_prices = price_data[close_col].values[-5:]
        cache_key = f'momentum_{hash(tuple(last_prices))}'
        cache_hit, cached_result = self.cache_manager.get(cache_key)
        if cache_hit:
            return cached_result
        momentum = self._calculate_momentum(price_data)
        self.cache_manager.set(cache_key, momentum)
        return momentum

    def _calculate_momentum(self, price_data: pd.DataFrame) ->Optional[float]:
        """
        Calculate momentum for a price series with vectorized operations.

        Args:
            price_data: Price DataFrame

        Returns:
            Momentum value (-1.0 to 1.0) or None if calculation fails
        """
        close_col = next((col for col in price_data.columns if col.lower() in
            ['close', 'price', 'adj close']), None)
        if not close_col or len(price_data) < 14:
            return None
        close_prices = price_data[close_col].values
        if len(close_prices) == 0 or np.isnan(close_prices[-1]):
            return None
        n_periods = min(14, len(close_prices) - 1)
        roc = (close_prices[-1] / close_prices[-n_periods - 1] - 1
            ) * 100 if close_prices[-n_periods - 1] > 0 else 0
        ema12 = self._calculate_ema(close_prices, 12)
        ema26 = self._calculate_ema(close_prices, 26)
        macd = ema12 - ema26 if ema12 is not None and ema26 is not None else 0
        rsi = self._calculate_rsi_vectorized(close_prices)
        norm_roc = np.clip(roc / 10, -1, 1)
        norm_macd = np.clip(macd * 2, -1, 1) if macd != 0 else 0
        norm_rsi = (rsi - 50) / 50 if rsi is not None else 0
        momentum = 0.4 * norm_roc + 0.3 * norm_macd + 0.3 * norm_rsi
        return momentum

    def _calculate_rsi_vectorized(self, prices: np.ndarray, period: int=14
        ) ->Optional[float]:
        """
        Calculate RSI using vectorized operations.

        Args:
            prices: Price array
            period: RSI period

        Returns:
            RSI value or None if calculation fails
        """
        if len(prices) <= period:
            return None
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
        return rsi

    def _calculate_ema(self, prices: np.ndarray, period: int) ->Optional[float
        ]:
        """
        Calculate EMA using vectorized operations.

        Args:
            prices: Price array
            period: EMA period

        Returns:
            EMA value or None if calculation fails
        """
        if len(prices) < period:
            return None
        return pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1]

    def _analyze_pair_divergence_optimized(self, related_pair: str,
        correlation: float, price_data: pd.DataFrame, primary_momentum: float
        ) ->Optional[Dict[str, Any]]:
        """
        Optimized divergence analysis for a single related pair.

        Args:
            related_pair: The related currency pair
            correlation: Correlation value with primary pair
            price_data: Price DataFrame for the related pair
            primary_momentum: Momentum value for the primary pair

        Returns:
            Dictionary with divergence details or None if no divergence
        """
        related_momentum = self._calculate_momentum_cached(price_data)
        if related_momentum is None:
            return None
        expected_momentum = primary_momentum * correlation
        momentum_diff = related_momentum - expected_momentum
        divergence_threshold = 0.5 * (1.0 - 0.5 * abs(correlation))
        if abs(momentum_diff) <= divergence_threshold:
            return None
        divergence_strength = min(1.0, abs(momentum_diff) / (1.0 + 0.5 *
            abs(correlation)))
        return {'pair': related_pair, 'correlation': correlation,
            'primary_momentum': primary_momentum, 'related_momentum':
            related_momentum, 'expected_momentum': expected_momentum,
            'momentum_difference': momentum_diff, 'divergence_type': 
            'positive' if momentum_diff > 0 else 'negative',
            'divergence_strength': divergence_strength}
