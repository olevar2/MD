"""
Advanced Calculation Techniques Module

This module implements advanced calculation techniques for indicator processing,
including pre-aggregation, incremental calculation, and smart caching strategies.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Optional, Tuple, Union, Set
import logging
from functools import lru_cache
import hashlib
import pickle
import time
from datetime import datetime, timedelta
import warnings
logger = logging.getLogger(__name__)


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class IncrementalCalculator:
    """
    Implements incremental calculation techniques for technical indicators.
    
    This class allows for efficient updates to indicator values when new data
    arrives, without recalculating the entire dataset.
    """

    def __init__(self):
        """Initialize the incremental calculator."""
        self.state_registry = {}
        self.functions = {}

    def register_incremental_function(self, name: str, init_func: Callable,
        update_func: Callable, finalize_func: Optional[Callable]=None):
        """
        Register an incremental calculation function.
        
        Args:
            name: Name of the function
            init_func: Function to initialize calculation state
            update_func: Function to update state with new data
            finalize_func: Function to finalize calculation (optional)
        """
        self.functions[name] = {'init': init_func, 'update': update_func,
            'finalize': finalize_func}
        logger.info(f'Registered incremental function: {name}')

    def create_calculation(self, func_name: str, calc_id: str, **kwargs) ->str:
        """
        Create a new incremental calculation state.
        
        Args:
            func_name: Name of the registered function
            calc_id: Identifier for this calculation instance
            **kwargs: Arguments for the init function
            
        Returns:
            Calculation ID
        """
        if func_name not in self.functions:
            raise ValueError(f'Function {func_name} not registered')
        state = self.functions[func_name]['init'](**kwargs)
        self.state_registry[calc_id] = {'func_name': func_name, 'state':
            state, 'last_update': datetime.now(), 'kwargs': kwargs}
        return calc_id

    def update_calculation(self, calc_id: str, new_data: Any) ->Any:
        """
        Update an incremental calculation with new data.
        
        Args:
            calc_id: Calculation ID
            new_data: New data to update with
            
        Returns:
            Updated result
        """
        if calc_id not in self.state_registry:
            raise ValueError(f'Calculation {calc_id} not found')
        calc_info = self.state_registry[calc_id]
        func_name = calc_info['func_name']
        state = calc_info['state']
        update_func = self.functions[func_name]['update']
        result = update_func(state, new_data)
        self.state_registry[calc_id]['last_update'] = datetime.now()
        return result

    def finalize_calculation(self, calc_id: str) ->Any:
        """
        Finalize an incremental calculation.
        
        Args:
            calc_id: Calculation ID
            
        Returns:
            Final result
        """
        if calc_id not in self.state_registry:
            raise ValueError(f'Calculation {calc_id} not found')
        calc_info = self.state_registry[calc_id]
        func_name = calc_info['func_name']
        state = calc_info['state']
        if (finalize_func := self.functions[func_name].get('finalize')
            ) is not None:
            result = finalize_func(state)
        else:
            result = state
        return result

    def remove_calculation(self, calc_id: str):
        """
        Remove a calculation state.
        
        Args:
            calc_id: Calculation ID to remove
        """
        if calc_id in self.state_registry:
            del self.state_registry[calc_id]
            logger.debug(f'Removed calculation state for {calc_id}')

    def cleanup_stale_calculations(self, max_age_hours: int=24):
        """
        Clean up stale calculation states.
        
        Args:
            max_age_hours: Maximum age in hours for a calculation state
        """
        now = datetime.now()
        to_remove = []
        for calc_id, info in self.state_registry.items():
            age = now - info['last_update']
            if age > timedelta(hours=max_age_hours):
                to_remove.append(calc_id)
        for calc_id in to_remove:
            self.remove_calculation(calc_id)
        if to_remove:
            logger.info(f'Cleaned up {len(to_remove)} stale calculation states'
                )


class SmartCache:
    """
    Smart caching system for optimizing indicator calculations.
    
    This class implements intelligent caching strategies for indicator values,
    allowing for reuse of calculations and minimizing redundant processing.
    """

    def __init__(self, max_size_mb: int=1000, ttl_seconds: int=3600):
        """
        Initialize the smart cache.
        
        Args:
            max_size_mb: Maximum cache size in MB
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.size_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) ->str:
        """
        Generate a unique key for the function call.
        
        Args:
            func_name: Function name
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            String key
        """
        key_parts = [func_name]
        for arg in args:
            if isinstance(arg, (pd.DataFrame, pd.Series)):
                arg_hash = hashlib.md5(pd.util.hash_pandas_object(arg).values
                    ).hexdigest()
                key_parts.append(f'df:{arg_hash}')
            elif isinstance(arg, np.ndarray):
                arg_hash = hashlib.md5(arg.tobytes()).hexdigest()
                key_parts.append(f'np:{arg_hash}')
            else:
                key_parts.append(str(arg))
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (pd.DataFrame, pd.Series)):
                v_hash = hashlib.md5(pd.util.hash_pandas_object(v).values
                    ).hexdigest()
                key_parts.append(f'{k}:df:{v_hash}')
            elif isinstance(v, np.ndarray):
                v_hash = hashlib.md5(v.tobytes()).hexdigest()
                key_parts.append(f'{k}:np:{v_hash}')
            else:
                key_parts.append(f'{k}:{v}')
        return hashlib.md5(':'.join(key_parts).encode()).hexdigest()

    def get(self, func_name: str, args: tuple, kwargs: dict) ->Tuple[bool, Any
        ]:
        """
        Get a value from the cache.
        
        Args:
            func_name: Function name
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            Tuple of (hit, value)
        """
        key = self._generate_key(func_name, args, kwargs)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] <= self.ttl_seconds:
                self.hits += 1
                entry['timestamp'] = time.time()
                return True, entry['value']
            else:
                del self.cache[key]
                self.size_bytes -= entry['size_bytes']
        self.misses += 1
        return False, None

    @with_exception_handling
    def set(self, func_name: str, args: tuple, kwargs: dict, value: Any):
        """
        Set a value in the cache.
        
        Args:
            func_name: Function name
            args: Function arguments
            kwargs: Function keyword arguments
            value: Value to cache
        """
        key = self._generate_key(func_name, args, kwargs)
        try:
            value_bytes = pickle.dumps(value)
            size_bytes = len(value_bytes)
        except Exception:
            logger.warning(
                f'Could not cache value for {func_name} (not serializable)')
            return
        if size_bytes > self.max_size_bytes * 0.25:
            logger.warning(
                f'Value for {func_name} is too large to cache ({size_bytes / 1024 / 1024:.2f} MB)'
                )
            return
        while (self.size_bytes + size_bytes > self.max_size_bytes and self.
            cache):
            oldest_key = min(self.cache.items(), key=lambda x: x[1][
                'timestamp'])[0]
            oldest_entry = self.cache.pop(oldest_key)
            self.size_bytes -= oldest_entry['size_bytes']
            self.evictions += 1
        self.cache[key] = {'value': value, 'timestamp': time.time(),
            'size_bytes': size_bytes}
        self.size_bytes += size_bytes

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.size_bytes = 0

    def get_stats(self) ->dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        return {'size_mb': self.size_bytes / 1024 / 1024, 'max_size_mb': 
            self.max_size_bytes / 1024 / 1024, 'usage_percent': self.
            size_bytes / self.max_size_bytes * 100 if self.max_size_bytes >
            0 else 0, 'entries': len(self.cache), 'hits': self.hits,
            'misses': self.misses, 'hit_rate': hit_rate * 100, 'evictions':
            self.evictions}


class PreAggregator:
    """
    Pre-aggregation system for optimizing indicator calculations.
    
    This class handles pre-aggregation of time series data for different timeframes,
    reducing redundant calculations when computing indicators across multiple timeframes.
    """

    def __init__(self):
        """Initialize the pre-aggregator."""
        self.aggregated_data = {}
        self.aggregation_specs = {}

    def register_aggregation(self, name: str, source_timeframe: str,
        target_timeframes: List[str], aggregation_funcs: Dict[str, str]):
        """
        Register an aggregation specification.
        
        Args:
            name: Aggregation name
            source_timeframe: Source data timeframe
            target_timeframes: List of target timeframes
            aggregation_funcs: Dict of column to aggregation function
        """
        self.aggregation_specs[name] = {'source_timeframe':
            source_timeframe, 'target_timeframes': target_timeframes,
            'aggregation_funcs': aggregation_funcs}
        for timeframe in target_timeframes:
            key = f'{name}_{timeframe}'
            if key not in self.aggregated_data:
                self.aggregated_data[key] = None
        logger.info(
            f"Registered aggregation: {name} ({source_timeframe} → {', '.join(target_timeframes)})"
            )

    def aggregate(self, name: str, data: pd.DataFrame) ->Dict[str, pd.DataFrame
        ]:
        """
        Perform aggregation on the input data.
        
        Args:
            name: Aggregation name
            data: Source data DataFrame with datetime index
            
        Returns:
            Dictionary of timeframe to aggregated DataFrame
        """
        if name not in self.aggregation_specs:
            raise ValueError(f'Aggregation {name} not registered')
        spec = self.aggregation_specs[name]
        result = {}
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'datetime' in data.columns:
                data = data.set_index('datetime')
            else:
                raise ValueError(
                    "Data must have DatetimeIndex or 'datetime' column")
        for timeframe in spec['target_timeframes']:
            resample_rule = self._timeframe_to_resample_rule(timeframe)
            resampled = data.resample(resample_rule)
            aggregated = resampled.agg(spec['aggregation_funcs'])
            key = f'{name}_{timeframe}'
            self.aggregated_data[key] = aggregated
            result[timeframe] = aggregated
        return result

    def get_aggregated_data(self, name: str, timeframe: str) ->Optional[pd.
        DataFrame]:
        """
        Get pre-aggregated data.
        
        Args:
            name: Aggregation name
            timeframe: Target timeframe
            
        Returns:
            Aggregated DataFrame or None if not available
        """
        key = f'{name}_{timeframe}'
        return self.aggregated_data.get(key, None)

    def update_aggregated_data(self, name: str, new_data: pd.DataFrame):
        """
        Update pre-aggregated data with new source data.
        
        Args:
            name: Aggregation name
            new_data: New source data
            
        Returns:
            Dictionary of updated aggregations
        """
        return self.aggregate(name, new_data)

    def _timeframe_to_resample_rule(self, timeframe: str) ->str:
        """
        Convert a timeframe string to a pandas resample rule.
        
        Args:
            timeframe: Timeframe string (e.g., '1m', '5m', '1h', '4h', '1d')
            
        Returns:
            Pandas resample rule
        """
        timeframe = timeframe.lower()
        if timeframe.endswith('min') or timeframe.endswith('m'):
            if timeframe.endswith('min'):
                num = int(timeframe[:-3])
                return f'{num}min'
            else:
                num = int(timeframe[:-1])
                return f'{num}min'
        elif timeframe.endswith('h'):
            num = int(timeframe[:-1])
            return f'{num}H'
        elif timeframe.endswith('d'):
            num = int(timeframe[:-1])
            return f'{num}D'
        elif timeframe.endswith('w'):
            num = int(timeframe[:-1])
            return f'{num}W'
        elif timeframe.endswith('mo'):
            num = int(timeframe[:-2])
            return f'{num}M'
        else:
            raise ValueError(f'Unsupported timeframe format: {timeframe}')


class LazyCalculator:
    """
    Lazy computation system for indicators.
    
    This class implements lazy evaluation of indicator calculations,
    performing calculations only when results are actually needed.
    """

    def __init__(self):
        """Initialize the lazy calculator."""
        self.computation_graph = {}
        self.cached_results = {}

    def register_computation(self, name: str, compute_func: Callable,
        dependencies: List[str]=None):
        """
        Register a computation node in the graph.
        
        Args:
            name: Node name
            compute_func: Computation function
            dependencies: List of dependent node names
        """
        self.computation_graph[name] = {'func': compute_func,
            'dependencies': dependencies or []}
        logger.debug(
            f'Registered computation: {name} (depends on: {dependencies})')

    def invalidate(self, name: str):
        """
        Invalidate a cached result and all dependent results.
        
        Args:
            name: Node name to invalidate
        """
        to_invalidate = set([name])

        def find_dependents(node):
    """
    Find dependents.
    
    Args:
        node: Description of node
    
    """

            for n, info in self.computation_graph.items():
                if node in info['dependencies'] and n not in to_invalidate:
                    to_invalidate.add(n)
                    find_dependents(n)
        find_dependents(name)
        for node in to_invalidate:
            if node in self.cached_results:
                del self.cached_results[node]
                logger.debug(f'Invalidated cached result: {node}')

    def compute(self, name: str, *args, **kwargs) ->Any:
        """
        Compute a result, using cached values when available.
        
        Args:
            name: Node name
            *args: Additional arguments for the compute function
            **kwargs: Additional keyword arguments
            
        Returns:
            Computation result
        """
        if name not in self.computation_graph:
            raise ValueError(f'Computation {name} not registered')
        if name in self.cached_results:
            return self.cached_results[name]
        node_info = self.computation_graph[name]
        dep_results = {}
        for dep in node_info['dependencies']:
            dep_results[dep] = self.compute(dep, *args, **kwargs)
        func = node_info['func']
        result = func(*args, dep_results=dep_results, **kwargs)
        self.cached_results[name] = result
        return result

    def clear_cache(self):
        """Clear all cached results."""
        self.cached_results.clear()
        logger.debug('Cleared computation cache')


incremental_calculator = IncrementalCalculator()
smart_cache = SmartCache()
pre_aggregator = PreAggregator()
lazy_calculator = LazyCalculator()


def register_incremental_ma(name: str='moving_average'):
    """
    Register an incremental moving average calculation.
    
    Args:
        name: Name for the registration
    """

    def init_ma(window: int, **kwargs):
    """
    Init ma.
    
    Args:
        window: Description of window
        kwargs: Description of kwargs
    
    """

        return {'window': window, 'values': [], 'sum': 0.0}

    def update_ma(state: dict, new_value: float) ->float:
    """
    Update ma.
    
    Args:
        state: Description of state
        new_value: Description of new_value
    
    Returns:
        float: Description of return value
    
    """

        values = state['values']
        current_sum = state['sum']
        window = state['window']
        values.append(new_value)
        current_sum += new_value
        if len(values) > window:
            removed = values.pop(0)
            current_sum -= removed
        state['sum'] = current_sum
        return current_sum / len(values)
    incremental_calculator.register_incremental_function(name, init_ma,
        update_ma)
    logger.info(f"Registered incremental moving average as '{name}'")


def register_incremental_rsi(name: str='rsi'):
    """
    Register an incremental RSI calculation.
    
    Args:
        name: Name for the registration
    """

    def init_rsi(window: int=14, **kwargs):
    """
    Init rsi.
    
    Args:
        window: Description of window
        kwargs: Description of kwargs
    
    """

        return {'window': window, 'values': [], 'gains': [], 'losses': [],
            'avg_gain': None, 'avg_loss': None, 'is_initialized': False}

    def update_rsi(state: dict, new_value: float) ->float:
    """
    Update rsi.
    
    Args:
        state: Description of state
        new_value: Description of new_value
    
    Returns:
        float: Description of return value
    
    """

        values = state['values']
        window = state['window']
        gains = state['gains']
        losses = state['losses']
        if len(values) > 0:
            change = new_value - values[-1]
            gain = max(0, change)
            loss = max(0, -change)
            gains.append(gain)
            losses.append(loss)
        values.append(new_value)
        if len(values) <= window:
            return 50.0
        if len(values) > window + 1:
            values.pop(0)
            gains.pop(0)
            losses.pop(0)
        if not state['is_initialized'] and len(gains) == window:
            state['avg_gain'] = sum(gains) / window
            state['avg_loss'] = sum(losses) / window
            state['is_initialized'] = True
        elif state['is_initialized']:
            state['avg_gain'] = (state['avg_gain'] * (window - 1) + gains[-1]
                ) / window
            state['avg_loss'] = (state['avg_loss'] * (window - 1) + losses[-1]
                ) / window
        else:
            return 50.0
        if state['avg_loss'] == 0:
            return 100.0
        rs = state['avg_gain'] / state['avg_loss']
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return rsi
    incremental_calculator.register_incremental_function(name, init_rsi,
        update_rsi)
    logger.info(f"Registered incremental RSI as '{name}'")


def register_common_aggregations():
    """Register common pre-aggregations for OHLCV data."""
    ohlcv_agg_funcs = {'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'}
    pre_aggregator.register_aggregation(name='ohlcv_1m', source_timeframe=
        '1m', target_timeframes=['5m', '15m', '30m', '1h', '4h', '1d'],
        aggregation_funcs=ohlcv_agg_funcs)
    pre_aggregator.register_aggregation(name='ohlcv_5m', source_timeframe=
        '5m', target_timeframes=['15m', '30m', '1h', '4h', '1d'],
        aggregation_funcs=ohlcv_agg_funcs)
    logger.info('Registered common OHLCV aggregations')


def initialize_optimization():
    """Initialize all optimization components."""
    register_incremental_ma()
    register_incremental_rsi()
    register_common_aggregations()
    logger.info('Optimization components initialized')
