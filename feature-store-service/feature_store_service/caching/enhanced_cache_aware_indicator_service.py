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


class EnhancedCacheAwareIndicatorService:
    """
    Enhanced indicator service with advanced caching strategies.
    
    This service extends the base CacheAwareIndicatorService with:
    1. Intermediate calculation caching to avoid redundant operations
    2. Support for predictive cache warming
    3. Adaptive TTL based on market volatility
    4. Degraded mode support for high-load conditions
    """
    
    def __init__(self, cache_manager: EnhancedCacheManager, indicator_factory, config: Dict[str, Any] = None):
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
        
        # Configure degraded mode settings
        self.enable_degraded_mode = self.config.get('enable_degraded_mode', True)
        self.load_threshold = self.config.get('degraded_mode_load_threshold', 0.85)
        self.memory_threshold = self.config.get('degraded_mode_memory_threshold', 0.9)
        self.rapid_request_threshold = self.config.get('rapid_request_threshold', 50)  # requests per second
        
        # Performance monitoring
        self.request_timestamps = []
        self.last_load_check = time.time()
        self.current_load_level = 0.0
        self.in_degraded_mode = False
        
        self.logger = logging.getLogger(__name__)
        
    async def calculate_indicator(
        self,
        indicator_type: str,
        params: Dict[str, Any],
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        use_degraded_mode: bool = None  # Override global setting
    ) -> pd.DataFrame:
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
        # Extract time range from data
        start_time = data.index.min()
        end_time = data.index.max()
        
        # Create cache key for the full calculation
        cache_key = CacheKey(
            indicator_type=indicator_type,
            params=params,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        # Monitor request rate for load management
        self._record_request()
        
        # Check if degraded mode should be used
        use_degraded = self._should_use_degraded_mode() if use_degraded_mode is None else use_degraded_mode
        
        # Try to get from cache first
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Calculate indicator
        start_calc_time = time.time()
        
        try:
            # Create indicator instance
            indicator = self.indicator_factory.create(indicator_type, **params)
            
            # Check if indicator supports degraded mode and we should use it
            if use_degraded and hasattr(indicator, 'calculate_degraded'):
                self.logger.info(f"Using degraded mode for {indicator_type} calculation")
                result = await self._calculate_with_degraded_mode(indicator, data, cache_key, params)
            else:
                # Use enhanced calculation with intermediate caching
                result = await self._calculate_with_intermediate_caching(indicator, data, cache_key, params)
                
            calc_time = time.time() - start_calc_time
            
            # Update request statistics with calculation time
            self._update_calculation_stats(indicator_type, calc_time)
            
            # Store in cache with adaptive TTL
            await self.cache_manager.set(cache_key, result)
            
            # Update volatility metrics for adaptive TTL
            self.cache_manager.update_price_data(symbol, data)
            
            # Trigger predictive cache warming for related indicators
            asyncio.create_task(self._trigger_cache_warming(indicator_type, symbol, timeframe, params))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating indicator {indicator_type}: {str(e)}")
            raise
        
    async def calculate_batch(
        self,
        indicator_configs: List[Dict[str, Any]],
        data: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> pd.DataFrame:
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
            
        # Update volatility metrics for adaptive TTL
        self.cache_manager.update_price_data(symbol, data)
        
        # Sort indicators by priority and dependencies
        sorted_configs = self._sort_indicators_for_batch(indicator_configs)
        
        # Create result DataFrame starting with input data
        result = data.copy()
        
        # Process indicators one by one, potentially using intermediate results
        for config in sorted_configs:
            indicator_type = config['indicator_type']
            params = config.get('params', {})
            
            # Create indicator-specific key
            sub_result = await self.calculate_indicator(
                indicator_type=indicator_type,
                params=params,
                data=result.copy(),  # Use accumulated results so far
                symbol=symbol,
                timeframe=timeframe
            )
            
            # Add new columns to result
            for col in sub_result.columns:
                if col not in result.columns:
                    result[col] = sub_result[col]
        
        return result
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = await self.cache_manager.get_stats()
        stats.update({
            "degraded_mode_active": self.in_degraded_mode,
            "current_load_level": self.current_load_level,
            "request_rate": self._calculate_request_rate()
        })
        return stats
    
    async def clear_cache_for_symbol(self, symbol: str) -> int:
        """
        Clear cache entries for a specific symbol.
        
        Args:
            symbol: Symbol to clear cache for
            
        Returns:
            Number of entries cleared
        """
        pattern = f"*:{symbol}:*"
        return await self.cache_manager.invalidate_pattern(pattern)
    
    async def _calculate_with_intermediate_caching(
        self,
        indicator,
        data: pd.DataFrame,
        cache_key: CacheKey,
        params: Dict[str, Any]
    ) -> pd.DataFrame:
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
        # Check if indicator supports intermediate calculations
        if not hasattr(indicator, 'get_intermediate_steps'):
            # Standard calculation without intermediate steps
            return indicator.calculate(data)
            
        # Get intermediate calculation steps
        intermediate_steps = indicator.get_intermediate_steps()
        
        if not intermediate_steps:
            # No intermediate steps defined
            return indicator.calculate(data)
            
        # Process each intermediate step with caching
        intermediate_results = {}
        
        for step in intermediate_steps:
            step_name = step['name']
            step_deps = step.get('depends_on', [])
            
            # Check if we have this intermediate result cached
            step_result = await self.cache_manager.get_intermediate_result(cache_key, step_name)
            
            if step_result is None:
                # Calculate this step
                if hasattr(indicator, f"calculate_{step_name}"):
                    # Get dependent results
                    step_inputs = {
                        dep: intermediate_results.get(dep)
                        for dep in step_deps
                        if dep in intermediate_results
                    }
                    
                    # Calculate this step
                    step_calc_func = getattr(indicator, f"calculate_{step_name}")
                    step_result = step_calc_func(data, **step_inputs, **params)
                    
                    # Cache the intermediate result
                    await self.cache_manager.set_intermediate_result(
                        cache_key,
                        step_name,
                        step_result,
                        step_deps
                    )
            
            # Store for potential use by subsequent steps
            intermediate_results[step_name] = step_result
            
        # Final calculation using intermediate results
        if hasattr(indicator, "calculate_with_intermediates"):
            return indicator.calculate_with_intermediates(data, intermediate_results, **params)
        else:
            # Fallback to standard calculation
            return indicator.calculate(data)
    
    async def _calculate_with_degraded_mode(
        self,
        indicator,
        data: pd.DataFrame,
        cache_key: CacheKey,
        params: Dict[str, Any]
    ) -> pd.DataFrame:
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
        # Determine degradation level based on load
        self.current_load_level = self._get_current_load()
        degradation_level = min(1.0, max(0.0, (self.current_load_level - 0.5) / 0.5))
        
        try:
            # Use degraded calculation method if available
            result = indicator.calculate_degraded(data, degradation_level=degradation_level, **params)
            
            # Mark the result as calculated in degraded mode
            if isinstance(result, pd.DataFrame) and hasattr(result, 'attrs'):
                result.attrs['degraded_mode'] = True
                result.attrs['degradation_level'] = degradation_level
                
            return result
        except Exception as e:
            self.logger.warning(f"Error in degraded mode calculation: {str(e)}. Falling back to standard calculation.")
            return indicator.calculate(data, **params)
    
    def _sort_indicators_for_batch(self, indicator_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort indicators for optimal batch processing order.
        
        Args:
            indicator_configs: List of indicator configurations
            
        Returns:
            Sorted list of indicator configurations
        """
        # Map indicator types to their dependencies
        dependencies = {}
        
        for config in indicator_configs:
            indicator_type = config['indicator_type']
            
            # Create indicator instance to check dependencies
            try:
                indicator = self.indicator_factory.create(
                    indicator_type, 
                    **(config.get('params', {}))
                )
                
                # Get dependencies if available
                if hasattr(indicator, 'get_dependencies'):
                    dependencies[indicator_type] = indicator.get_dependencies()
                else:
                    dependencies[indicator_type] = []
            except Exception as e:
                self.logger.warning(f"Error creating indicator {indicator_type}: {str(e)}")
                dependencies[indicator_type] = []
        
        # Perform topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(node):
            if node in temp_visited:
                # Cyclic dependency, break the cycle
                return
            if node in visited:
                return
                
            temp_visited.add(node)
            
            for dep in dependencies.get(node, []):
                # Only consider dependencies that are in our batch
                if dep in dependencies:
                    visit(dep)
                    
            temp_visited.remove(node)
            visited.add(node)
            order.append(node)
        
        # Visit all nodes
        for indicator_type in dependencies:
            if indicator_type not in visited:
                visit(indicator_type)
        
        # Sort configs based on the order
        type_to_config = {
            config['indicator_type']: config
            for config in indicator_configs
        }
        
        sorted_configs = [
            type_to_config[indicator_type]
            for indicator_type in order
            if indicator_type in type_to_config
        ]
        
        return sorted_configs
    
    def _record_request(self) -> None:
        """Record a new request for load monitoring."""
        now = time.time()
        
        # Add current timestamp
        self.request_timestamps.append(now)
        
        # Remove timestamps older than 10 seconds
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts <= 10]
    
    def _calculate_request_rate(self) -> float:
        """
        Calculate current request rate (requests per second).
        
        Returns:
            Requests per second over the last 10 seconds
        """
        now = time.time()
        recent_requests = [ts for ts in self.request_timestamps if now - ts <= 10]
        
        if not recent_requests:
            return 0.0
        
        # Calculate requests per second
        oldest = min(recent_requests)
        timespan = max(1.0, now - oldest)  # Avoid division by zero
        return len(recent_requests) / timespan
    
    def _should_use_degraded_mode(self) -> bool:
        """
        Determine if degraded mode should be used based on system load.
        
        Returns:
            True if degraded mode should be activated
        """
        if not self.enable_degraded_mode:
            return False
            
        # Check if we're already in degraded mode
        if self.in_degraded_mode:
            # Stay in degraded mode until load drops well below the threshold
            if self._get_current_load() < self.load_threshold * 0.8:
                self.in_degraded_mode = False
                self.logger.info("Exiting degraded mode")
                return False
            return True
            
        # Check if we should enter degraded mode
        if self._get_current_load() > self.load_threshold:
            self.in_degraded_mode = True
            self.logger.warning("Entering degraded mode due to high load")
            return True
            
        # Check request rate
        if self._calculate_request_rate() > self.rapid_request_threshold:
            self.in_degraded_mode = True
            self.logger.warning(f"Entering degraded mode due to high request rate: {self._calculate_request_rate():.1f}/s")
            return True
            
        return False
    
    def _get_current_load(self) -> float:
        """
        Get current system load level (0.0 to 1.0).
        
        Returns:
            Load level from 0.0 (idle) to 1.0 (fully loaded)
        """
        # Check at most once per second
        now = time.time()
        if now - self.last_load_check < 1.0:
            return self.current_load_level
            
        self.last_load_check = now
        
        try:
            import psutil
            # Average of CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
            memory_percent = psutil.virtual_memory().percent / 100.0
            
            self.current_load_level = (cpu_percent + memory_percent) / 2
            return self.current_load_level
            
        except ImportError:
            # If psutil is not available, estimate based on request rate
            request_rate = self._calculate_request_rate()
            self.current_load_level = min(1.0, request_rate / self.rapid_request_threshold)
            return self.current_load_level
    
    async def _trigger_cache_warming(
        self,
        indicator_type: str,
        symbol: str,
        timeframe: str,
        params: Dict[str, Any]
    ) -> None:
        """
        Trigger predictive cache warming for related indicators.
        
        Args:
            indicator_type: Type of the current indicator
            symbol: Symbol identifier
            timeframe: Timeframe identifier
            params: Current indicator parameters
        """
        # Skip if predictive warming is disabled
        if not getattr(self.cache_manager, 'enable_predictive_warming', False):
            return
            
        # Define related indicators that might be needed soon
        related_indicators = []
        
        # Same indicator type with different parameters
        variations = self._get_indicator_variations(indicator_type, params)
        for var_params in variations:
            related_indicators.append({
                'indicator_type': indicator_type,
                'symbol': symbol,
                'timeframe': timeframe,
                'params': var_params
            })
            
        # Related indicator types that are commonly used together
        related_types = self._get_related_indicator_types(indicator_type)
        for rel_type in related_types:
            related_indicators.append({
                'indicator_type': rel_type,
                'symbol': symbol,
                'timeframe': timeframe,
                'params': {}  # Use default parameters
            })
            
        # Trigger cache warming asynchronously
        if related_indicators:
            asyncio.create_task(self.cache_manager.warm_cache(
                related_indicators,
                self._get_data_for_symbol
            ))
            
    def _get_indicator_variations(self, indicator_type: str, current_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get variations of parameters for an indicator type.
        
        Args:
            indicator_type: Type of indicator
            current_params: Current parameters
            
        Returns:
            List of parameter dictionaries for variations
        """
        variations = []
        
        # Different parameter values based on common patterns
        if 'period' in current_params:
            period = current_params['period']
            variations.append({**current_params, 'period': period * 2})
            variations.append({**current_params, 'period': period // 2})
            
        if 'window' in current_params:
            window = current_params['window']
            variations.append({**current_params, 'window': window * 2})
            variations.append({**current_params, 'window': window // 2})
            
        return variations
        
    def _get_related_indicator_types(self, indicator_type: str) -> List[str]:
        """
        Get related indicator types commonly used together.
        
        Args:
            indicator_type: Type of indicator
            
        Returns:
            List of related indicator types
        """
        # Define common groupings of related indicators
        indicator_groups = {
            'sma': ['ema', 'wma', 'dema', 'tema'],
            'ema': ['sma', 'macd', 'ppo'],
            'rsi': ['stoch', 'stochrsi', 'cci'],
            'macd': ['ema', 'ppo', 'adx'],
            'bollinger': ['atr', 'keltner', 'donchian'],
            'atr': ['bollinger', 'keltner', 'chandelier'],
        }
        
        # Normalize indicator type (remove parameters)
        base_type = indicator_type.lower().split('_')[0]
        
        # Return related types
        return indicator_groups.get(base_type, [])
        
    async def _get_data_for_symbol(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get data for a symbol and timeframe for cache warming.
        
        Args:
            symbol: Symbol identifier
            timeframe: Timeframe identifier
            
        Returns:
            DataFrame with price data or None if not available
        """
        # In a real implementation, this would access your data provider
        # For now, we'll return None as a placeholder
        return None
