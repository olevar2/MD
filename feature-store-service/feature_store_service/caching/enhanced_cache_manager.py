"""
Enhanced Cache Manager

This module extends the base CacheManager with advanced features:
- Intermediate calculation caching
- Predictive cache warming
- Adaptive TTL based on data volatility
"""
import os
import json
import logging
import threading
import pickle
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from .cache_key import CacheKey
from .cache_metrics import CacheMetrics
from .memory_cache import LRUCache
from .disk_cache import DiskCache


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class EnhancedCacheManager:
    """
    Enhanced cache manager with advanced caching strategies for technical indicators.

    This class extends the base cache manager with:
    1. Intermediate calculation caching for improved performance
    2. Predictive cache warming based on usage patterns
    3. Adaptive TTL strategy based on data volatility
    """

    def __init__(self, config: Dict[str, Any]=None, cache_dir: str='cache',
        max_memory_size: int=1024, default_ttl: int=3600,
        enable_intermediate_caching: bool=True, enable_predictive_warming:
        bool=True, enable_adaptive_ttl: bool=True, volatility_sensitivity:
        float=0.5):
        """
        Initialize enhanced cache manager.

        Args:
            config: Configuration dictionary
            cache_dir: Directory for disk cache
            max_memory_size: Maximum memory cache size in MB
            default_ttl: Default time-to-live for cache entries in seconds
            enable_intermediate_caching: Whether to cache intermediate calculations
            enable_predictive_warming: Whether to enable predictive cache warming
            enable_adaptive_ttl: Whether to enable adaptive TTL based on data volatility
            volatility_sensitivity: Sensitivity to data volatility for TTL adjustment (0-1)
        """
        self.config = config or {}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_memory_size = max_memory_size
        self.default_ttl = timedelta(seconds=default_ttl)
        self.enable_intermediate_caching = enable_intermediate_caching
        self.enable_predictive_warming = enable_predictive_warming
        self.enable_adaptive_ttl = enable_adaptive_ttl
        self.volatility_sensitivity = max(0, min(1, volatility_sensitivity))
        self.memory_cache = LRUCache(max_size_mb=max_memory_size)
        self.disk_cache = DiskCache(cache_dir=str(self.cache_dir))
        self.metrics = CacheMetrics()
        self.access_history = defaultdict(list)
        self.symbol_timeframe_patterns = defaultdict(Counter)
        self.indicator_param_patterns = defaultdict(Counter)
        self.dependency_graph = {}
        self.intermediate_results = {}
        self.price_history = {}
        self.volatility_metrics = {}
        self.ttl_adjustments = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    async def get(self, key: Union[str, CacheKey]) ->Optional[Any]:
        """
        Get item from cache.

        Args:
            key: Cache key (string or CacheKey object)

        Returns:
            Cached item or None if not found
        """
        cache_key = str(key) if isinstance(key, CacheKey) else key
        if self.enable_predictive_warming:
            self._record_access(key)
        memory_result = self.memory_cache.get(cache_key)
        if memory_result is not None:
            self.metrics.record_hit(CacheMetrics.Level.MEMORY)
            return memory_result
        disk_result = await self.disk_cache.get(cache_key)
        if disk_result is not None:
            self.memory_cache.set(cache_key, disk_result)
            self.metrics.record_hit(CacheMetrics.Level.DISK)
            return disk_result
        self.metrics.record_miss()
        return None

    async def set(self, key: Union[str, CacheKey], value: Any, ttl:
        Optional[int]=None) ->None:
        """
        Store item in cache with optional custom TTL.

        Args:
            key: Cache key (string or CacheKey object)
            value: Value to cache
            ttl: Time-to-live in seconds (overrides default and adaptive TTL)
        """
        cache_key = str(key) if isinstance(key, CacheKey) else key
        if self.enable_adaptive_ttl and ttl is None:
            ttl_seconds = self._calculate_adaptive_ttl(key)
        else:
            ttl_seconds = ttl if ttl is not None else int(self.default_ttl.
                total_seconds())
        self.memory_cache.set(cache_key, value, ttl=ttl_seconds)
        await self.disk_cache.set(cache_key, value, ttl=ttl_seconds)

    async def set_intermediate_result(self, key: Union[str, CacheKey],
        intermediate_key: str, value: Any, depends_on_keys: Optional[List[
        str]]=None) ->None:
        """
        Store intermediate calculation result with dependency tracking.

        Args:
            key: Main cache key (string or CacheKey object)
            intermediate_key: Key for the intermediate result
            value: Intermediate calculation value
            depends_on_keys: Optional list of keys this calculation depends on
        """
        if not self.enable_intermediate_caching:
            return
        main_key = str(key) if isinstance(key, CacheKey) else key
        combined_key = f'{main_key}:intermediate:{intermediate_key}'
        self.intermediate_results[combined_key] = value
        if main_key not in self.dependency_graph:
            self.dependency_graph[main_key] = set()
        self.dependency_graph[main_key].add(combined_key)
        if depends_on_keys:
            for dep_key in depends_on_keys:
                if dep_key in self.dependency_graph:
                    self.dependency_graph[main_key].update(self.
                        dependency_graph[dep_key])

    async def get_intermediate_result(self, key: Union[str, CacheKey],
        intermediate_key: str) ->Optional[Any]:
        """
        Retrieve an intermediate calculation result.

        Args:
            key: Main cache key (string or CacheKey object)
            intermediate_key: Key for the intermediate result

        Returns:
            Intermediate result or None if not found
        """
        if not self.enable_intermediate_caching:
            return None
        main_key = str(key) if isinstance(key, CacheKey) else key
        combined_key = f'{main_key}:intermediate:{intermediate_key}'
        return self.intermediate_results.get(combined_key)

    async def invalidate(self, key: Union[str, CacheKey]) ->None:
        """
        Invalidate a cache entry and its dependent intermediate calculations.

        Args:
            key: Cache key to invalidate
        """
        cache_key = str(key) if isinstance(key, CacheKey) else key
        self.memory_cache.delete(cache_key)
        await self.disk_cache.delete(cache_key)
        if cache_key in self.dependency_graph:
            for intermediate_key in self.dependency_graph[cache_key]:
                if intermediate_key in self.intermediate_results:
                    del self.intermediate_results[intermediate_key]
            del self.dependency_graph[cache_key]
        self.metrics.record_invalidation()

    async def invalidate_pattern(self, pattern: str) ->int:
        """
        Invalidate all cache entries matching a pattern.

        Args:
            pattern: Pattern to match against cache keys

        Returns:
            Number of invalidated entries
        """
        memory_keys = self.memory_cache.find_keys(pattern)
        disk_keys = await self.disk_cache.find_keys(pattern)
        all_keys = set(memory_keys) | set(disk_keys)
        count = 0
        for key in all_keys:
            await self.invalidate(key)
            count += 1
        return count

    async def prefetch(self, key: Union[str, CacheKey], calculator_func, *
        args, **kwargs) ->Any:
        """
        Fetch or calculate data with caching.

        Args:
            key: Cache key
            calculator_func: Function to calculate data if not in cache
            *args, **kwargs: Arguments for calculator_func

        Returns:
            Cached or freshly calculated data
        """
        result = await self.get(key)
        if result is not None:
            return result
        result = await calculator_func(*args, **kwargs)
        await self.set(key, result)
        return result

    async def get_stats(self) ->Dict[str, Any]:
        """
        Get cache statistics and metrics.

        Returns:
            Dictionary with cache statistics
        """
        memory_stats = self.memory_cache.get_stats()
        disk_stats = await self.disk_cache.get_stats()
        metrics = self.metrics.get_metrics()
        return {'memory_cache': memory_stats, 'disk_cache': disk_stats,
            'metrics': metrics, 'intermediate_cache_size': len(self.
            intermediate_results), 'predictive_warming_patterns': len(self.
            symbol_timeframe_patterns) + len(self.indicator_param_patterns),
            'adaptive_ttl_adjustments': len(self.ttl_adjustments)}

    async def clear(self) ->int:
        """
        Clear all cache entries.

        Returns:
            Number of cleared entries
        """
        memory_count = self.memory_cache.clear()
        disk_count = await self.disk_cache.clear()
        intermediate_count = len(self.intermediate_results)
        self.intermediate_results.clear()
        self.dependency_graph.clear()
        self.access_history.clear()
        self.symbol_timeframe_patterns.clear()
        self.indicator_param_patterns.clear()
        self.price_history.clear()
        self.volatility_metrics.clear()
        self.ttl_adjustments.clear()
        total_count = memory_count + disk_count + intermediate_count
        self.logger.info(f'Cleared {total_count} cache entries')
        return total_count

    @async_with_exception_handling
    async def warm_cache(self, indicator_specs: List[Dict[str, Any]],
        data_provider) ->int:
        """
        Predictively warm the cache for likely-to-be-used indicators.

        Args:
            indicator_specs: List of indicator specifications
            data_provider: Function to provide data for calculations

        Returns:
            Number of warmed cache entries
        """
        if not self.enable_predictive_warming:
            return 0
        warmed_count = 0
        indicators_to_warm = self._predict_indicators_to_warm(indicator_specs)
        for spec in indicators_to_warm:
            try:
                symbol = spec.get('symbol')
                timeframe = spec.get('timeframe')
                indicator_type = spec.get('indicator_type')
                params = spec.get('params', {})
                key = CacheKey(indicator_type=indicator_type, params=params,
                    symbol=symbol, timeframe=timeframe, start_time=None,
                    end_time=None)
                if await self.get(key) is not None:
                    continue
                data = await data_provider(symbol, timeframe)
                if data is None or data.empty:
                    continue
                key.start_time = data.index.min()
                key.end_time = data.index.max()
                await self.set(key, {'warmed': True, 'time': datetime.now()})
                warmed_count += 1
            except Exception as e:
                self.logger.error(f'Error warming cache for {spec}: {e}')
        return warmed_count

    def update_price_data(self, symbol: str, data: pd.DataFrame) ->None:
        """
        Update price data for volatility calculation.

        Args:
            symbol: Symbol identifier
            data: Price data DataFrame
        """
        if not self.enable_adaptive_ttl:
            return
        with self._lock:
            self.price_history[symbol] = data
            if 'close' in data.columns and len(data) >= 20:
                returns = data['close'].pct_change().dropna()
                if len(returns) >= 10:
                    self.volatility_metrics[symbol] = returns.std()

    def _calculate_adaptive_ttl(self, key: Union[str, CacheKey]) ->int:
        """
        Calculate adaptive TTL based on data volatility.

        Args:
            key: Cache key

        Returns:
            TTL in seconds
        """
        if not self.enable_adaptive_ttl:
            return int(self.default_ttl.total_seconds())
        base_ttl = int(self.default_ttl.total_seconds())
        symbol = key.symbol if isinstance(key, CacheKey) else None
        if symbol is None or symbol not in self.volatility_metrics:
            return base_ttl
        volatility = self.volatility_metrics[symbol]
        mean_volatility = 0.01
        relative_volatility = volatility / mean_volatility
        adjustment_factor = np.exp(-self.volatility_sensitivity * (
            relative_volatility - 1))
        adjustment_factor = max(0.1, min(2.0, adjustment_factor))
        self.ttl_adjustments[str(key)] = adjustment_factor
        return int(base_ttl * adjustment_factor)

    def _record_access(self, key: Union[str, CacheKey]) ->None:
        """
        Record access pattern for predictive warming.

        Args:
            key: Accessed cache key
        """
        if not self.enable_predictive_warming:
            return
        with self._lock:
            str_key = str(key)
            now = datetime.now()
            self.access_history[str_key].append(now)
            if len(self.access_history[str_key]) > 100:
                self.access_history[str_key] = self.access_history[str_key][
                    -100:]
            if isinstance(key, CacheKey):
                if key.symbol and key.timeframe:
                    pattern_key = key.symbol, key.timeframe
                    self.symbol_timeframe_patterns[pattern_key][key.
                        indicator_type] += 1
                if key.indicator_type:
                    params_tuple = tuple(sorted([f'{k}={v}' for k, v in key
                        .params.items()]))
                    self.indicator_param_patterns[key.indicator_type][
                        params_tuple] += 1

    @with_exception_handling
    def _predict_indicators_to_warm(self, all_indicators: List[Dict[str, Any]]
        ) ->List[Dict[str, Any]]:
        """
        Predict which indicators should be pre-warmed based on access patterns.

        Args:
            all_indicators: List of all available indicator specifications

        Returns:
            List of indicator specifications predicted to be used soon
        """
        if not self.enable_predictive_warming:
            return []
        predicted_indicators = []
        recent_cutoff = datetime.now() - timedelta(hours=2)
        recent_accesses = {}
        for key, timestamps in self.access_history.items():
            recent = [ts for ts in timestamps if ts > recent_cutoff]
            if recent:
                recent_accesses[key] = len(recent)
        frequent_keys = sorted(recent_accesses.keys(), key=lambda k:
            recent_accesses[k], reverse=True)
        top_keys = frequent_keys[:max(1, len(frequent_keys) // 5)]
        for key_str in top_keys:
            try:
                if ':' in key_str:
                    parts = key_str.split(':')
                    if len(parts) >= 3:
                        indicator_type = parts[0]
                        symbol = parts[1]
                        timeframe = parts[2]
                        if (symbol, timeframe
                            ) in self.symbol_timeframe_patterns:
                            common_indicators = self.symbol_timeframe_patterns[
                                symbol, timeframe]
                            for ind_type, count in common_indicators.most_common(
                                3):
                                if ind_type in self.indicator_param_patterns:
                                    for params_tuple, _ in self.indicator_param_patterns[
                                        ind_type].most_common(2):
                                        params = {}
                                        for param_str in params_tuple:
                                            if '=' in param_str:
                                                k, v = param_str.split('=', 1)
                                                params[k] = v
                                        predicted_indicators.append({
                                            'indicator_type': ind_type,
                                            'symbol': symbol, 'timeframe':
                                            timeframe, 'params': params})
            except Exception as e:
                self.logger.error(
                    f'Error predicting indicators from key {key_str}: {e}')
        for spec in all_indicators:
            symbol = spec.get('symbol')
            timeframe = spec.get('timeframe')
            indicator_type = spec.get('indicator_type')
            if (symbol, timeframe) in self.symbol_timeframe_patterns:
                frequent_indicators = self.symbol_timeframe_patterns[symbol,
                    timeframe]
                if (indicator_type in frequent_indicators and 
                    frequent_indicators[indicator_type] >= 3):
                    if spec not in predicted_indicators:
                        predicted_indicators.append(spec)
        return predicted_indicators[:20]
