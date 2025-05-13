"""
Enhanced Cache-Aware Indicator Service

This module provides an enhanced cache-aware service for technical indicators,
leveraging advanced caching features to optimize performance.
"""
import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from datetime import datetime
import pandas as pd
import numpy as np
from .enhanced_cache_manager import EnhancedCacheManager
from .cache_key import CacheKey


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class EnhancedCacheAwareIndicatorService:
    """
    Enhanced indicator service with advanced caching strategies.
    
    This service extends the base CacheAwareIndicatorService with:
    1. Intermediate calculation caching to avoid redundant operations
    2. Support for predictive cache warming
    3. Adaptive TTL based on market volatility
    4. Degraded mode support for high-load conditions
    """

    def __init__(self, cache_manager: EnhancedCacheManager,
        indicator_factory, config: Dict[str, Any]=None):
        """
        Initialize the enhanced cache-aware indicator service.
        
        Args:
            cache_manager: Enhanced cache manager instance
            indicator_factory: Factory to create indicator instances
            config: Optional configuration parameters
        """
        self.cache_manager = cache_manager
        self.indicator_factory = indicator_factory
        self.config = config or {}
        self.enable_degraded_mode = self.config.get('enable_degraded_mode',
            True)
        self.load_threshold = self.config.get('degraded_mode_load_threshold',
            0.85)
        self.memory_threshold = self.config.get(
            'degraded_mode_memory_threshold', 0.9)
        self.rapid_request_threshold = self.config.get(
            'rapid_request_threshold', 50)
        self.request_timestamps = []
        self.last_load_check = time.time()
        self.current_load_level = 0.0
        self.in_degraded_mode = False
        self.logger = logging.getLogger(__name__)

    @async_with_exception_handling
    async def calculate_indicator(self, indicator_type: str, params: Dict[
        str, Any], data: pd.DataFrame, symbol: str, timeframe: str,
        use_degraded_mode: bool=None) ->pd.DataFrame:
        """
        Calculate indicator with enhanced caching support.
        
        Args:
            indicator_type: Type of indicator to calculate
            params: Parameters for the indicator
            data: Input data for calculation
            symbol: Symbol for the data
            timeframe: Timeframe for the data
            use_degraded_mode: Override to force or disable degraded mode
            
        Returns:
            DataFrame with indicator results
        """
        start_time = data.index.min()
        end_time = data.index.max()
        cache_key = CacheKey(indicator_type=indicator_type, params=params,
            symbol=symbol, timeframe=timeframe, start_time=start_time,
            end_time=end_time)
        self._record_request()
        use_degraded = self._should_use_degraded_mode(
            ) if use_degraded_mode is None else use_degraded_mode
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result is not None:
            return cached_result
        start_calc_time = time.time()
        try:
            indicator = self.indicator_factory.create(indicator_type, **params)
            if use_degraded and hasattr(indicator, 'calculate_degraded'):
                self.logger.info(
                    f'Using degraded mode for {indicator_type} calculation')
                result = await self._calculate_with_degraded_mode(indicator,
                    data, cache_key, params)
            else:
                result = await self._calculate_with_intermediate_caching(
                    indicator, data, cache_key, params)
            calc_time = time.time() - start_calc_time
            self._update_calculation_stats(indicator_type, calc_time)
            await self.cache_manager.set(cache_key, result)
            self.cache_manager.update_price_data(symbol, data)
            asyncio.create_task(self._trigger_cache_warming(indicator_type,
                symbol, timeframe, params))
            return result
        except Exception as e:
            self.logger.error(
                f'Error calculating indicator {indicator_type}: {str(e)}')
            raise

    async def calculate_batch(self, indicator_configs: List[Dict[str, Any]],
        data: pd.DataFrame, symbol: str, timeframe: str) ->pd.DataFrame:
        """
        Calculate multiple indicators in batch with optimized caching.
        
        Args:
            indicator_configs: List of indicator configurations
            data: Input data for all calculations
            symbol: Symbol for the data
            timeframe: Timeframe for the data
            
        Returns:
            DataFrame with all indicator results combined
        """
        if not indicator_configs:
            return data.copy()
        self.cache_manager.update_price_data(symbol, data)
        sorted_configs = self._sort_indicators_for_batch(indicator_configs)
        result = data.copy()
        for config in sorted_configs:
            indicator_type = config['indicator_type']
            params = config_manager.get('params', {})
            sub_result = await self.calculate_indicator(indicator_type=
                indicator_type, params=params, data=result.copy(), symbol=
                symbol, timeframe=timeframe)
            for col in sub_result.columns:
                if col not in result.columns:
                    result[col] = sub_result[col]
        return result

    async def get_cache_stats(self) ->Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = await self.cache_manager.get_stats()
        stats.update({'degraded_mode_active': self.in_degraded_mode,
            'current_load_level': self.current_load_level, 'request_rate':
            self._calculate_request_rate()})
        return stats

    async def clear_cache_for_symbol(self, symbol: str) ->int:
        """
        Clear cache entries for a specific symbol.
        
        Args:
            symbol: Symbol to clear cache for
            
        Returns:
            Number of entries cleared
        """
        pattern = f'*:{symbol}:*'
        return await self.cache_manager.invalidate_pattern(pattern)

    async def _calculate_with_intermediate_caching(self, indicator, data:
        pd.DataFrame, cache_key: CacheKey, params: Dict[str, Any]
        ) ->pd.DataFrame:
        """
        Calculate indicator with intermediate result caching.
        
        Args:
            indicator: Indicator instance
            data: Input data
            cache_key: Cache key for the full calculation
            params: Indicator parameters
            
        Returns:
            Calculated indicator result
        """
        if not hasattr(indicator, 'get_intermediate_steps'):
            return indicator.calculate(data)
        intermediate_steps = indicator.get_intermediate_steps()
        if not intermediate_steps:
            return indicator.calculate(data)
        intermediate_results = {}
        for step in intermediate_steps:
            step_name = step['name']
            step_deps = step.get('depends_on', [])
            step_result = await self.cache_manager.get_intermediate_result(
                cache_key, step_name)
            if step_result is None:
                if hasattr(indicator, f'calculate_{step_name}'):
                    step_inputs = {dep: intermediate_results.get(dep) for
                        dep in step_deps if dep in intermediate_results}
                    step_calc_func = getattr(indicator,
                        f'calculate_{step_name}')
                    step_result = step_calc_func(data, **step_inputs, **params)
                    await self.cache_manager.set_intermediate_result(cache_key,
                        step_name, step_result, step_deps)
            intermediate_results[step_name] = step_result
        if hasattr(indicator, 'calculate_with_intermediates'):
            return indicator.calculate_with_intermediates(data,
                intermediate_results, **params)
        else:
            return indicator.calculate(data)

    @async_with_exception_handling
    async def _calculate_with_degraded_mode(self, indicator, data: pd.
        DataFrame, cache_key: CacheKey, params: Dict[str, Any]) ->pd.DataFrame:
        """
        Calculate indicator using degraded mode for performance.
        
        Args:
            indicator: Indicator instance
            data: Input data
            cache_key: Cache key
            params: Indicator parameters
            
        Returns:
            Calculated result in degraded mode
        """
        self.current_load_level = self._get_current_load()
        degradation_level = min(1.0, max(0.0, (self.current_load_level - 
            0.5) / 0.5))
        try:
            result = indicator.calculate_degraded(data, degradation_level=
                degradation_level, **params)
            if isinstance(result, pd.DataFrame) and hasattr(result, 'attrs'):
                result.attrs['degraded_mode'] = True
                result.attrs['degradation_level'] = degradation_level
            return result
        except Exception as e:
            self.logger.warning(
                f'Error in degraded mode calculation: {str(e)}. Falling back to standard calculation.'
                )
            return indicator.calculate(data, **params)

    @with_exception_handling
    def _sort_indicators_for_batch(self, indicator_configs: List[Dict[str,
        Any]]) ->List[Dict[str, Any]]:
        """
        Sort indicators for optimal batch processing order.
        
        Args:
            indicator_configs: List of indicator configurations
            
        Returns:
            Sorted list of indicator configurations
        """
        dependencies = {}
        for config in indicator_configs:
            indicator_type = config['indicator_type']
            try:
                indicator = self.indicator_factory.create(indicator_type,
                    **config_manager.get('params', {}))
                if hasattr(indicator, 'get_dependencies'):
                    dependencies[indicator_type] = indicator.get_dependencies()
                else:
                    dependencies[indicator_type] = []
            except Exception as e:
                self.logger.warning(
                    f'Error creating indicator {indicator_type}: {str(e)}')
                dependencies[indicator_type] = []
        visited = set()
        temp_visited = set()
        order = []

        def visit(node):
    """
    Visit.
    
    Args:
        node: Description of node
    
    """

            if node in temp_visited:
                return
            if node in visited:
                return
            temp_visited.add(node)
            for dep in dependencies.get(node, []):
                if dep in dependencies:
                    visit(dep)
            temp_visited.remove(node)
            visited.add(node)
            order.append(node)
        for indicator_type in dependencies:
            if indicator_type not in visited:
                visit(indicator_type)
        type_to_config = {config['indicator_type']: config for config in
            indicator_configs}
        sorted_configs = [type_to_config[indicator_type] for indicator_type in
            order if indicator_type in type_to_config]
        return sorted_configs

    def _record_request(self) ->None:
        """Record a new request for load monitoring."""
        now = time.time()
        self.request_timestamps.append(now)
        self.request_timestamps = [ts for ts in self.request_timestamps if 
            now - ts <= 10]

    def _calculate_request_rate(self) ->float:
        """
        Calculate current request rate (requests per second).
        
        Returns:
            Requests per second over the last 10 seconds
        """
        now = time.time()
        recent_requests = [ts for ts in self.request_timestamps if now - ts <=
            10]
        if not recent_requests:
            return 0.0
        oldest = min(recent_requests)
        timespan = max(1.0, now - oldest)
        return len(recent_requests) / timespan

    def _should_use_degraded_mode(self) ->bool:
        """
        Determine if degraded mode should be used based on system load.
        
        Returns:
            True if degraded mode should be activated
        """
        if not self.enable_degraded_mode:
            return False
        if self.in_degraded_mode:
            if self._get_current_load() < self.load_threshold * 0.8:
                self.in_degraded_mode = False
                self.logger.info('Exiting degraded mode')
                return False
            return True
        if self._get_current_load() > self.load_threshold:
            self.in_degraded_mode = True
            self.logger.warning('Entering degraded mode due to high load')
            return True
        if self._calculate_request_rate() > self.rapid_request_threshold:
            self.in_degraded_mode = True
            self.logger.warning(
                f'Entering degraded mode due to high request rate: {self._calculate_request_rate():.1f}/s'
                )
            return True
        return False

    @with_exception_handling
    def _get_current_load(self) ->float:
        """
        Get current system load level (0.0 to 1.0).
        
        Returns:
            Load level from 0.0 (idle) to 1.0 (fully loaded)
        """
        now = time.time()
        if now - self.last_load_check < 1.0:
            return self.current_load_level
        self.last_load_check = now
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
            memory_percent = psutil.virtual_memory().percent / 100.0
            self.current_load_level = (cpu_percent + memory_percent) / 2
            return self.current_load_level
        except ImportError:
            request_rate = self._calculate_request_rate()
            self.current_load_level = min(1.0, request_rate / self.
                rapid_request_threshold)
            return self.current_load_level

    async def _trigger_cache_warming(self, indicator_type: str, symbol: str,
        timeframe: str, params: Dict[str, Any]) ->None:
        """
        Trigger predictive cache warming for related indicators.
        
        Args:
            indicator_type: Type of the current indicator
            symbol: Symbol identifier
            timeframe: Timeframe identifier
            params: Current indicator parameters
        """
        if not getattr(self.cache_manager, 'enable_predictive_warming', False):
            return
        related_indicators = []
        variations = self._get_indicator_variations(indicator_type, params)
        for var_params in variations:
            related_indicators.append({'indicator_type': indicator_type,
                'symbol': symbol, 'timeframe': timeframe, 'params': var_params}
                )
        related_types = self._get_related_indicator_types(indicator_type)
        for rel_type in related_types:
            related_indicators.append({'indicator_type': rel_type, 'symbol':
                symbol, 'timeframe': timeframe, 'params': {}})
        if related_indicators:
            asyncio.create_task(self.cache_manager.warm_cache(
                related_indicators, self._get_data_for_symbol))

    def _get_indicator_variations(self, indicator_type: str, current_params:
        Dict[str, Any]) ->List[Dict[str, Any]]:
        """
        Get variations of parameters for an indicator type.
        
        Args:
            indicator_type: Type of indicator
            current_params: Current parameters
            
        Returns:
            List of parameter dictionaries for variations
        """
        variations = []
        if 'period' in current_params:
            period = current_params['period']
            variations.append({**current_params, 'period': period * 2})
            variations.append({**current_params, 'period': period // 2})
        if 'window' in current_params:
            window = current_params['window']
            variations.append({**current_params, 'window': window * 2})
            variations.append({**current_params, 'window': window // 2})
        return variations

    def _get_related_indicator_types(self, indicator_type: str) ->List[str]:
        """
        Get related indicator types commonly used together.
        
        Args:
            indicator_type: Type of indicator
            
        Returns:
            List of related indicator types
        """
        indicator_groups = {'sma': ['ema', 'wma', 'dema', 'tema'], 'ema': [
            'sma', 'macd', 'ppo'], 'rsi': ['stoch', 'stochrsi', 'cci'],
            'macd': ['ema', 'ppo', 'adx'], 'bollinger': ['atr', 'keltner',
            'donchian'], 'atr': ['bollinger', 'keltner', 'chandelier']}
        base_type = indicator_type.lower().split('_')[0]
        return indicator_groups.get(base_type, [])

    async def _get_data_for_symbol(self, symbol: str, timeframe: str
        ) ->Optional[pd.DataFrame]:
        """
        Get data for a symbol and timeframe for cache warming.
        
        Args:
            symbol: Symbol identifier
            timeframe: Timeframe identifier
            
        Returns:
            DataFrame with price data or None if not available
        """
        return None
