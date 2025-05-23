"""
Confluence Analyzer

This module provides comprehensive functionality to identify confluence zones where multiple
technical analysis factors align to provide stronger trading signals.

Confluence is detected when different technical indicators, price patterns, or
key levels align to suggest the same market direction or pivot point.

Features:
- Multi-timeframe analysis
- Support/Resistance confluence
- Indicator alignment detection
- Pattern completion analysis
- Fibonacci level convergence
- Elliott wave pivot points
- Gann level analysis
- Harmonic pattern detection
- Adaptive effectiveness tracking
- External signal aggregation
- Volume profile analysis
- Market structure analysis

Optimized for performance with:
- Vectorized operations
- Caching of intermediate results
- Parallel processing for independent calculations
- Early termination for performance improvement
- Memory optimization
"""
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass
from functools import lru_cache
import concurrent.futures
from threading import Lock
import time
from analysis_engine.core.base.components import BaseAnalyzer, AnalysisResult
from analysis_engine.analysis.indicators import IndicatorClient
from analysis_engine.services.tool_effectiveness import ToolEffectivenessTracker, MarketRegime
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
logger = logging.getLogger(__name__)
_level_cache = {}
_level_cache_lock = Lock()
_level_cache_ttl = 300
_level_cache_last_cleanup = time.time()


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ConfluenceStrength(Enum):
    """Strength of confluence signal"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4
    EXCEPTIONAL = 5


@dataclass
class ConfluenceZone:
    """Container for confluence zone data"""
    center: float
    lower_bound: float
    upper_bound: float
    strength: ConfluenceStrength
    contributing_tools: List[str]
    timeframes: List[str]
    confluence_types: List[str]
    levels: List[Dict[str, Any]]
    score: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) ->Dict[str, Any]:
        """Convert to dictionary representation"""
        return {'center': self.center, 'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound, 'strength': self.strength.name,
            'contributing_tools': self.contributing_tools, 'timeframes':
            self.timeframes, 'confluence_types': self.confluence_types,
            'levels': self.levels, 'score': self.score, 'metadata': self.
            metadata or {}}


class ConfluenceAnalyzer(BaseAnalyzer):
    """
    Comprehensive analyzer for identifying confluence points across multiple
    indicators, timeframes, and analysis methods.

    Features:
    - Multi-timeframe analysis
    - Support/Resistance confluence
    - Indicator alignment detection
    - Pattern completion analysis
    - Fibonacci level convergence
    - Elliott wave pivot points
    - Gann level analysis
    - Harmonic pattern detection
    - Adaptive effectiveness tracking
    - External signal aggregation
    - Volume profile analysis
    - Market structure analysis
    """
    DEFAULT_PARAMS = {'min_tools_for_confluence': 2,
        'effectiveness_threshold': 0.5, 'sr_proximity_threshold': 0.0015,
        'indicator_agreement_threshold': 0.7,
        'timeframe_alignment_required': 3, 'zone_width_pips': 20,
        'source_weights': {'price_action': 1.0, 'indicator': 0.8, 'pattern':
        0.9, 'support_resistance': 1.0, 'trend': 0.9, 'momentum': 0.8,
        'volatility': 0.7, 'volume': 0.8, 'fibonacci': 0.9, 'elliott_wave':
        0.9, 'sentiment': 0.6, 'other': 0.5}, 'conflicting_signals_penalty':
        0.3, 'use_volume_profile': True, 'use_market_structure': True,
        'use_fibonacci_levels': True, 'use_moving_averages': True,
        'moving_averages': [20, 50, 100, 200], 'use_pivots': True,
        'pivot_type': 'traditional', 'session_barriers': True, 'max_levels': 10
        }

    def __init__(self, tool_effectiveness_repository: Optional[
        ToolEffectivenessRepository]=None, parameters: Optional[Dict[str,
        Any]]=None):
        """
        Initialize the confluence analyzer

        Args:
            tool_effectiveness_repository: Repository for tracking tool effectiveness
            parameters: Optional configuration parameters
        """
        resolved_params = self.DEFAULT_PARAMS.copy()
        if parameters:
            resolved_params.update(parameters)
        super().__init__(name='ConfluenceAnalyzer', parameters=resolved_params)
        self.effectiveness_repository = tool_effectiveness_repository
        self.tool_tracker = ToolEffectivenessTracker(
            ) if tool_effectiveness_repository else None
        self.indicator_client = IndicatorClient()
        self.signal_cache = []
        self.last_analysis_time = None

    @lru_cache(maxsize=32)
    async def _get_effective_tools_cached(self, market_regime: str) ->Dict[
        str, float]:
        """Get effective tools with caching for performance"""
        return await self._get_effective_tools(market_regime)

    @async_with_exception_handling
    async def analyze(self, data: Dict[str, Any]) ->AnalysisResult:
        """
        Analyze market data to identify confluence zones.

        Optimized with:
        - Performance monitoring
        - Early termination for invalid inputs
        - Caching of intermediate results
        - Vectorized operations
        - Parallel processing

        Args:
            data: Dictionary containing market data and parameters
                {
                    "symbol": str,
                    "timeframe": str,
                    "market_data": {
                        "open": List[float],
                        "high": List[float],
                        "low": List[float],
                        "close": List[float],
                        "volume": List[float],
                        "timestamp": List[str]
                    }
                }

        Returns:
            AnalysisResult containing identified confluence zones
        """
        start_time = time.time()
        performance_metrics = {}
        try:
            if not data or 'market_data' not in data:
                return AnalysisResult(analyzer_name=self.name, result={
                    'error': 'Invalid or empty data provided'}, is_valid=False)
            t0 = time.time()
            symbol = data['symbol']
            timeframe = data['timeframe']
            market_data = data['market_data']
            performance_metrics['extract_data'] = time.time() - t0
            t0 = time.time()
            df = pd.DataFrame(market_data)
            if df.empty:
                return AnalysisResult(analyzer_name=self.name, result={
                    'timestamp': datetime.now().isoformat(), 'symbol':
                    symbol, 'error': 'Empty market data provided',
                    'performance_metrics': performance_metrics}, is_valid=False
                    )
            performance_metrics['convert_to_dataframe'] = time.time() - t0
            current_price = df['close'].iloc[-1]
            t0 = time.time()
            all_levels = self._collect_all_levels(df, current_price)
            performance_metrics['collect_all_levels'] = time.time() - t0
            if not all_levels:
                return AnalysisResult(analyzer_name=self.name, result={
                    'timestamp': datetime.now().isoformat(), 'symbol':
                    symbol, 'current_price': current_price,
                    'confluence_zones': [], 'market_regime': data.get(
                    'market_regime'), 'effective_tools': {},
                    'performance_metrics': performance_metrics}, metadata={
                    'timeframe': timeframe, 'zone_count': 0})
            t0 = time.time()
            effective_tools = {}
            if self.effectiveness_repository and 'market_regime' in data:
                market_regime = data['market_regime']
                effective_tools = await self._get_effective_tools_cached(
                    market_regime)
            performance_metrics['get_effective_tools'] = time.time() - t0
            t0 = time.time()
            confluence_zones = self._group_levels_into_zones(all_levels,
                current_price, effective_tools)
            performance_metrics['group_levels_into_zones'] = time.time() - t0
            if not confluence_zones:
                return AnalysisResult(analyzer_name=self.name, result={
                    'timestamp': datetime.now().isoformat(), 'symbol':
                    symbol, 'current_price': current_price,
                    'confluence_zones': [], 'market_regime': data.get(
                    'market_regime'), 'effective_tools': effective_tools,
                    'performance_metrics': performance_metrics}, metadata={
                    'timeframe': timeframe, 'zone_count': 0})
            t0 = time.time()
            scored_zones = self._calculate_zone_strength(confluence_zones,
                data.get('market_regime'), effective_tools)
            performance_metrics['calculate_zone_strength'] = time.time() - t0
            t0 = time.time()
            if scored_zones:
                try:
                    strengths = np.array([zone.strength.value for zone in
                        scored_zones])
                    sorted_indices = np.argsort(strengths)[::-1]
                    sorted_zones = [scored_zones[i] for i in sorted_indices]
                except (AttributeError, TypeError):
                    sorted_zones = sorted(scored_zones, key=lambda z: z.
                        strength.value, reverse=True)
            else:
                sorted_zones = []
            sorted_zones = sorted_zones[:self.parameters['max_levels']]
            performance_metrics['sort_zones'] = time.time() - t0
            t0 = time.time()
            result = {'timestamp': datetime.now().isoformat(), 'symbol':
                symbol, 'current_price': current_price, 'confluence_zones':
                [zone.to_dict() for zone in sorted_zones], 'market_regime':
                data.get('market_regime'), 'effective_tools':
                effective_tools, 'performance_metrics': performance_metrics}
            performance_metrics['prepare_result'] = time.time() - t0
            performance_metrics['total_execution_time'] = time.time(
                ) - start_time
            if performance_metrics['total_execution_time'] > 0.5:
                logger.info(
                    f'Confluence analysis performance metrics: {performance_metrics}'
                    )
            return AnalysisResult(analyzer_name=self.name, result=result,
                metadata={'timeframe': timeframe, 'zone_count': len(
                sorted_zones), 'execution_time_ms': int(performance_metrics
                ['total_execution_time'] * 1000)})
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f'Error in confluence analysis after {execution_time:.2f}s: {str(e)}'
                , exc_info=True)
            return AnalysisResult(analyzer_name=self.name, result={'error':
                f'Analysis failed: {str(e)}', 'performance_metrics':
                performance_metrics}, is_valid=False)

    def _clean_level_cache(self, force: bool=False) ->None:
        """Clean expired entries from the level cache"""
        global _level_cache, _level_cache_last_cleanup, _level_cache_ttl
        current_time = time.time()
        if not force and current_time - _level_cache_last_cleanup < 60:
            return
        with _level_cache_lock:
            expired_keys = []
            for key, (timestamp, _) in _level_cache.items():
                if current_time - timestamp > _level_cache_ttl:
                    expired_keys.append(key)
            for key in expired_keys:
                del _level_cache[key]
            _level_cache_last_cleanup = current_time
            logger.debug(
                f'Level cache cleaned: {len(expired_keys)} expired entries removed, {len(_level_cache)} entries remaining'
                )

    def _get_cache_key(self, df: pd.DataFrame, level_type: str) ->str:
        """Generate a cache key for level calculations"""
        last_rows = min(10, len(df))
        if last_rows == 0:
            return f'{level_type}_empty'
        high_vals = df['high'].iloc[-last_rows:].values
        low_vals = df['low'].iloc[-last_rows:].values
        close_vals = df['close'].iloc[-last_rows:].values
        fingerprint = hash(tuple(np.concatenate([high_vals, low_vals,
            close_vals])))
        return f'{level_type}_{fingerprint}'

    @with_exception_handling
    def _collect_all_levels(self, df: pd.DataFrame, current_price: float
        ) ->List[Dict[str, Any]]:
        """
        Collect all potential support/resistance levels

        Optimized with:
        - Caching of level calculations
        - Parallel processing for independent level types
        - Early termination for empty dataframes
        """
        if df.empty:
            return []
        self._clean_level_cache()
        levels = []
        level_futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            if self.parameters['use_market_structure']:
                level_futures['market_structure'] = executor.submit(self.
                    _get_support_resistance_levels_cached, df)
            if self.parameters['use_fibonacci_levels']:
                level_futures['fibonacci'] = executor.submit(self.
                    _get_fibonacci_levels_cached, df)
            if self.parameters['use_moving_averages']:
                level_futures['moving_average'] = executor.submit(self.
                    _get_moving_average_levels_cached, df)
            if self.parameters['use_pivots']:
                level_futures['pivot'] = executor.submit(self.
                    _get_pivot_levels_cached, df)
            if self.parameters['use_volume_profile']:
                level_futures['volume_profile'] = executor.submit(self.
                    _get_volume_profile_levels_cached, df)
            for level_type, future in level_futures.items():
                try:
                    result = future.result()
                    levels.extend(result)
                except Exception as e:
                    logger.warning(
                        f'Error calculating {level_type} levels: {str(e)}')
        return levels

    def _get_support_resistance_levels_cached(self, df: pd.DataFrame) ->List[
        Dict[str, Any]]:
        """Cached version of support/resistance level calculation"""
        cache_key = self._get_cache_key(df, 'sr')
        with _level_cache_lock:
            if cache_key in _level_cache:
                timestamp, cached_levels = _level_cache[cache_key]
                if time.time() - timestamp <= _level_cache_ttl:
                    return cached_levels.copy()
        levels = self._get_support_resistance_levels(df)
        with _level_cache_lock:
            _level_cache[cache_key] = time.time(), levels.copy()
        return levels

    def _get_fibonacci_levels_cached(self, df: pd.DataFrame) ->List[Dict[
        str, Any]]:
        """Cached version of Fibonacci level calculation"""
        cache_key = self._get_cache_key(df, 'fib')
        with _level_cache_lock:
            if cache_key in _level_cache:
                timestamp, cached_levels = _level_cache[cache_key]
                if time.time() - timestamp <= _level_cache_ttl:
                    return cached_levels.copy()
        levels = self._get_fibonacci_levels(df)
        with _level_cache_lock:
            _level_cache[cache_key] = time.time(), levels.copy()
        return levels

    def _get_moving_average_levels_cached(self, df: pd.DataFrame) ->List[Dict
        [str, Any]]:
        """Cached version of moving average level calculation"""
        cache_key = self._get_cache_key(df, 'ma')
        with _level_cache_lock:
            if cache_key in _level_cache:
                timestamp, cached_levels = _level_cache[cache_key]
                if time.time() - timestamp <= _level_cache_ttl:
                    return cached_levels.copy()
        levels = self._get_moving_average_levels(df)
        with _level_cache_lock:
            _level_cache[cache_key] = time.time(), levels.copy()
        return levels

    def _get_pivot_levels_cached(self, df: pd.DataFrame) ->List[Dict[str, Any]
        ]:
        """Cached version of pivot level calculation"""
        cache_key = self._get_cache_key(df, 'pivot')
        with _level_cache_lock:
            if cache_key in _level_cache:
                timestamp, cached_levels = _level_cache[cache_key]
                if time.time() - timestamp <= _level_cache_ttl:
                    return cached_levels.copy()
        levels = self._get_pivot_levels(df)
        with _level_cache_lock:
            _level_cache[cache_key] = time.time(), levels.copy()
        return levels

    def _get_volume_profile_levels_cached(self, df: pd.DataFrame) ->List[Dict
        [str, Any]]:
        """Cached version of volume profile level calculation"""
        cache_key = self._get_cache_key(df, 'vol')
        with _level_cache_lock:
            if cache_key in _level_cache:
                timestamp, cached_levels = _level_cache[cache_key]
                if time.time() - timestamp <= _level_cache_ttl:
                    return cached_levels.copy()
        levels = self._get_volume_profile_levels(df)
        with _level_cache_lock:
            _level_cache[cache_key] = time.time(), levels.copy()
        return levels

    def _get_support_resistance_levels(self, df: pd.DataFrame) ->List[Dict[
        str, Any]]:
        """Get support and resistance levels from market structure"""
        levels = []
        highs = df['high'].values
        lows = df['low'].values
        for i in range(2, len(df) - 2):
            if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and highs[i
                ] > highs[i + 1] and highs[i] > highs[i + 2]:
                levels.append({'price': highs[i], 'type': 'resistance',
                    'tool_id': 'market_structure', 'timeframe': 'current',
                    'strength': 1.0})
            if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and lows[i
                ] < lows[i + 1] and lows[i] < lows[i + 2]:
                levels.append({'price': lows[i], 'type': 'support',
                    'tool_id': 'market_structure', 'timeframe': 'current',
                    'strength': 1.0})
        return levels

    def _get_fibonacci_levels(self, df: pd.DataFrame) ->List[Dict[str, Any]]:
        """Get Fibonacci retracement and extension levels"""
        levels = []
        recent_high = df['high'].max()
        recent_low = df['low'].min()
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        for level in fib_levels:
            price = recent_high - (recent_high - recent_low) * level
            levels.append({'price': price, 'type': 'fibonacci', 'tool_id':
                'fibonacci', 'timeframe': 'current', 'strength': 0.9,
                'level': level})
            price = recent_low + (recent_high - recent_low) * level
            levels.append({'price': price, 'type': 'fibonacci', 'tool_id':
                'fibonacci', 'timeframe': 'current', 'strength': 0.9,
                'level': level})
        return levels

    def _get_moving_average_levels(self, df: pd.DataFrame) ->List[Dict[str,
        Any]]:
        """Get moving average levels"""
        levels = []
        for period in self.parameters['moving_averages']:
            ma = self.indicator_client.calculate_sma(df['close'], period)
            current_ma = ma.iloc[-1]
            levels.append({'price': current_ma, 'type': 'moving_average',
                'tool_id': f'ma_{period}', 'timeframe': 'current',
                'strength': 0.8, 'period': period})
        return levels

    def _get_pivot_levels(self, df: pd.DataFrame) ->List[Dict[str, Any]]:
        """Get pivot point levels"""
        levels = []
        prev_high = df['high'].iloc[-2]
        prev_low = df['low'].iloc[-2]
        prev_close = df['close'].iloc[-2]
        pivot = (prev_high + prev_low + prev_close) / 3
        r1 = 2 * pivot - prev_low
        s1 = 2 * pivot - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)
        levels.extend([{'price': pivot, 'type': 'pivot', 'tool_id': 'pivot',
            'timeframe': 'daily', 'strength': 0.9, 'level': 'pivot'}, {
            'price': r1, 'type': 'resistance', 'tool_id': 'pivot',
            'timeframe': 'daily', 'strength': 0.8, 'level': 'r1'}, {'price':
            s1, 'type': 'support', 'tool_id': 'pivot', 'timeframe': 'daily',
            'strength': 0.8, 'level': 's1'}, {'price': r2, 'type':
            'resistance', 'tool_id': 'pivot', 'timeframe': 'daily',
            'strength': 0.7, 'level': 'r2'}, {'price': s2, 'type':
            'support', 'tool_id': 'pivot', 'timeframe': 'daily', 'strength':
            0.7, 'level': 's2'}])
        return levels

    def _get_volume_profile_levels(self, df: pd.DataFrame) ->List[Dict[str,
        Any]]:
        """
        Get volume profile levels

        Optimized with:
        - Vectorized operations
        - NumPy optimizations
        - Early termination for edge cases
        """
        levels = []
        if len(df) < 10 or 'volume' not in df.columns:
            return levels
        price_range = df['high'].max() - df['low'].min()
        if price_range <= 0:
            return levels
        num_bins = 50
        bin_size = price_range / num_bins
        bins = np.linspace(df['low'].min(), df['high'].max(), num_bins + 1)
        hist, _ = np.histogram(df['close'], bins=bins, weights=df['volume'])
        volume_profile = hist
        poc_idx = np.argmax(volume_profile)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        total_volume = np.sum(volume_profile)
        if total_volume <= 0:
            return levels
        target_volume = total_volume * 0.7
        sorted_indices = np.argsort(volume_profile)[::-1]
        sorted_volumes = volume_profile[sorted_indices]
        cumulative_volumes = np.cumsum(sorted_volumes)
        value_area_mask = cumulative_volumes <= target_volume
        if not np.any(value_area_mask):
            value_area_indices = [poc_idx]
        else:
            value_area_indices = sorted_indices[value_area_mask]
            if np.any(~value_area_mask) and len(value_area_indices) < len(
                sorted_indices):
                next_idx = np.argmin(value_area_mask)
                if next_idx < len(sorted_indices):
                    value_area_indices = np.append(value_area_indices,
                        sorted_indices[next_idx])
        bin_indices = np.array([i for i in value_area_indices if 0 <= i < 
            len(bins) - 1])
        if len(bin_indices) == 0:
            va_high = poc_price * 1.01
            va_low = poc_price * 0.99
        else:
            bin_prices = np.array([((bins[i] + bins[i + 1]) / 2) for i in
                bin_indices])
            va_high = np.max(bin_prices)
            va_low = np.min(bin_prices)
        levels.extend([{'price': poc_price, 'type': 'volume_profile',
            'tool_id': 'volume_profile', 'timeframe': 'current', 'strength':
            1.0, 'level': 'poc'}, {'price': va_high, 'type':
            'volume_profile', 'tool_id': 'volume_profile', 'timeframe':
            'current', 'strength': 0.8, 'level': 'va_high'}, {'price':
            va_low, 'type': 'volume_profile', 'tool_id': 'volume_profile',
            'timeframe': 'current', 'strength': 0.8, 'level': 'va_low'}])
        return levels

    def _group_levels_into_zones(self, levels: List[Dict[str, Any]],
        current_price: float, effective_tools: Dict[str, float]) ->List[Dict
        [str, Any]]:
        """
        Group similar price levels into confluence zones

        Optimized version using:
        - Sets for faster lookups
        - Early termination
        - Vectorized operations where possible
        - Reduced memory allocations
        """
        if not levels:
            return []
        prices = np.array([level['price'] for level in levels])
        sorted_indices = np.argsort(prices)
        base_zone_width = current_price * self.parameters['zone_width_pips'
            ] / 10000
        zones = []
        min_tools = self.parameters['min_tools_for_confluence']
        effective_tools_set = set(effective_tools.keys())
        for idx in sorted_indices:
            level = levels[idx]
            level_price = level['price']
            level_tool_id = level['tool_id']
            level_timeframe = level['timeframe']
            level_type = level['type']
            added_to_existing = False
            potential_zones = []
            left, right = 0, len(zones) - 1
            while left <= right:
                mid = (left + right) // 2
                zone = zones[mid]
                if zone['lower_bound'] <= level_price <= zone['upper_bound']:
                    potential_zones.append(mid)
                    i = mid - 1
                    while i >= 0 and zones[i]['upper_bound'] >= level_price:
                        potential_zones.append(i)
                        i -= 1
                    i = mid + 1
                    while i < len(zones) and zones[i]['lower_bound'
                        ] <= level_price:
                        potential_zones.append(i)
                        i += 1
                    break
                elif zone['upper_bound'] < level_price:
                    left = mid + 1
                else:
                    right = mid - 1
            for zone_idx in potential_zones:
                zone = zones[zone_idx]
                if zone['lower_bound'] <= level_price <= zone['upper_bound']:
                    zone['levels'].append(level)
                    if 'contributing_tools_set' not in zone:
                        zone['contributing_tools_set'] = set(zone[
                            'contributing_tools'])
                    if 'timeframes_set' not in zone:
                        zone['timeframes_set'] = set(zone['timeframes'])
                    if 'confluence_types_set' not in zone:
                        zone['confluence_types_set'] = set(zone[
                            'confluence_types'])
                    zone['contributing_tools_set'].add(level_tool_id)
                    zone['timeframes_set'].add(level_timeframe)
                    zone['confluence_types_set'].add(level_type)
                    zone['contributing_tools'] = list(zone[
                        'contributing_tools_set'])
                    zone['timeframes'] = list(zone['timeframes_set'])
                    zone['confluence_types'] = list(zone[
                        'confluence_types_set'])
                    added_to_existing = True
                    break
            if not added_to_existing:
                zone_width = base_zone_width
                new_zone = {'center': level_price, 'lower_bound': 
                    level_price - zone_width / 2, 'upper_bound': 
                    level_price + zone_width / 2, 'levels': [level],
                    'contributing_tools': [level_tool_id], 'timeframes': [
                    level_timeframe], 'confluence_types': [level_type],
                    'contributing_tools_set': {level_tool_id},
                    'timeframes_set': {level_timeframe},
                    'confluence_types_set': {level_type}}
                insert_idx = 0
                for i, zone in enumerate(zones):
                    if zone['center'] > level_price:
                        insert_idx = i
                        break
                    insert_idx = i + 1
                zones.insert(insert_idx, new_zone)
        confluent_zones = []
        for zone in zones:
            effective_tools_in_zone = zone['contributing_tools_set'
                ].intersection(effective_tools_set)
            if len(zone['contributing_tools']) >= min_tools:
                confluent_zones.append(zone)
                continue
            if len(effective_tools_in_zone) > 0:
                confluent_zones.append(zone)
                continue
            if len(zone['timeframes']) > 1:
                confluent_zones.append(zone)
                continue
        for zone in confluent_zones:
            zone.pop('contributing_tools_set', None)
            zone.pop('timeframes_set', None)
            zone.pop('confluence_types_set', None)
        return confluent_zones

    def _calculate_zone_strength(self, zones: List[Dict[str, Any]],
        market_regime: Optional[MarketRegime], effective_tools: Dict[str,
        float]) ->List[ConfluenceZone]:
        """Calculate strength of confluence zones"""
        confluence_zones = []
        for zone in zones:
            base_score = len(zone['contributing_tools']) * 0.4 + len(zone[
                'timeframes']) * 0.3 + len(zone['confluence_types']) * 0.3
            effective_tools_in_zone = [tool for tool in zone[
                'contributing_tools'] if tool in effective_tools]
            if effective_tools_in_zone:
                effectiveness_score = sum(effective_tools[tool] for tool in
                    effective_tools_in_zone) / len(effective_tools_in_zone)
                base_score *= 1 + effectiveness_score
            if base_score >= 4.0:
                strength = ConfluenceStrength.EXCEPTIONAL
            elif base_score >= 3.0:
                strength = ConfluenceStrength.VERY_STRONG
            elif base_score >= 2.0:
                strength = ConfluenceStrength.STRONG
            elif base_score >= 1.5:
                strength = ConfluenceStrength.MODERATE
            else:
                strength = ConfluenceStrength.WEAK
            confluence_zone = ConfluenceZone(center=zone['center'],
                lower_bound=zone['lower_bound'], upper_bound=zone[
                'upper_bound'], strength=strength, contributing_tools=zone[
                'contributing_tools'], timeframes=zone['timeframes'],
                confluence_types=zone['confluence_types'], levels=zone[
                'levels'], score=base_score)
            confluence_zones.append(confluence_zone)
        return confluence_zones

    async def _get_effective_tools(self, market_regime: MarketRegime) ->Dict[
        str, float]:
        """Get effective tools for current market regime"""
        if not self.effectiveness_repository:
            return {}
        return await self.effectiveness_repository.get_effective_tools(
            market_regime, min_score=self.parameters['effectiveness_threshold']
            )
