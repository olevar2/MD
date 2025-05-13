"""
Incremental Calculation Framework

This module provides optimized implementations for incrementally calculating
technical indicators with high performance and memory efficiency.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import heapq
from collections import deque


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class PerformanceMonitor:
    """
    Performance monitoring for critical calculation paths
    with detailed latency tracking and analysis
    """

    def __init__(self, capacity: int=1000):
        """
        Initialize performance monitor with a specified capacity for metrics storage
        
        Args:
            capacity: Maximum number of timing records to keep
        """
        self.logger = logging.getLogger(__name__)
        self.metrics = {}
        self.latency_records = {}
        self.capacity = capacity

    def record_latency(self, function_name: str, latency_ms: float) ->None:
        """
        Record the latency for a function call
        
        Args:
            function_name: Name of the function
            latency_ms: Execution time in milliseconds
        """
        if function_name not in self.metrics:
            self.metrics[function_name] = {'count': 0, 'total': 0.0, 'min':
                float('inf'), 'max': 0.0, 'average': 0.0, 'p95': 0.0, 'p99':
                0.0}
            self.latency_records[function_name] = deque(maxlen=self.capacity)
        metrics = self.metrics[function_name]
        metrics['count'] += 1
        metrics['total'] += latency_ms
        metrics['min'] = min(metrics['min'], latency_ms)
        metrics['max'] = max(metrics['max'], latency_ms)
        metrics['average'] = metrics['total'] / metrics['count']
        self.latency_records[function_name].append(latency_ms)
        if len(self.latency_records[function_name]) >= 10:
            records = sorted(self.latency_records[function_name])
            metrics['p95'] = records[int(0.95 * len(records))]
            metrics['p99'] = records[int(0.99 * len(records))]

    def get_all_metrics(self) ->Dict[str, Dict[str, Any]]:
        """Get all metrics for all monitored functions"""
        return self.metrics

    def get_function_metrics(self, function_name: str) ->Optional[Dict[str,
        Any]]:
        """Get metrics for a specific function"""
        return self.metrics.get(function_name)

    def get_slowest_functions(self, limit: int=5) ->List[Dict[str, Any]]:
        """Get the N slowest functions by average latency"""
        if not self.metrics:
            return []
        sorted_functions = sorted(self.metrics.items(), key=lambda item:
            item[1]['average'], reverse=True)
        result = []
        for name, metrics in sorted_functions[:limit]:
            result.append({'function': name, 'average_ms': metrics[
                'average'], 'p95_ms': metrics['p95'], 'call_count': metrics
                ['count']})
        return result

    def clear_metrics(self) ->None:
        """Reset all metrics data"""
        self.metrics = {}
        self.latency_records = {}

    def log_slow_operation(self, threshold_ms: float=100.0) ->None:
        """Log recently recorded operations that exceeded the threshold"""
        slow_ops = []
        for func_name, records in self.latency_records.items():
            recent_records = list(records)[-10:]
            for latency in recent_records:
                if latency > threshold_ms:
                    slow_ops.append((func_name, latency))
        if slow_ops:
            for func_name, latency in slow_ops:
                self.logger.warning(
                    f'Slow operation: {func_name} took {latency:.2f}ms')


performance_monitor = PerformanceMonitor()


@with_exception_handling
def measure_latency(func):
    """
    Decorator to measure and record function execution time
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function that measures execution time
    """

    @with_exception_handling
    def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            performance_monitor.record_latency(func.__name__, latency_ms)
    return wrapper


class IncrementalIndicator:
    """
    Base class for incrementally-calculated technical indicators
    with stateful updating and memory-efficient processing
    """

    def __init__(self, name: str, window_size: int, input_key: str='close'):
        """
        Initialize a new incremental indicator
        
        Args:
            name: Indicator name
            window_size: Size of data window for calculation
            input_key: Column name in input data to use for calculation
        """
        self.name = name
        self.window_size = window_size
        self.input_key = input_key
        self.state = {}
        self.values = []
        self.is_initialized = False
        self.logger = logging.getLogger(f'indicator.{name}')

    def initialize(self, data: Union[pd.DataFrame, np.ndarray]) ->np.ndarray:
        """
        Initialize indicator with historical data
        
        Args:
            data: Historical data in DataFrame or array format
            
        Returns:
            Calculated indicator values
        """
        self.is_initialized = True
        return self._calculate(data)

    def update(self, new_data_point) ->float:
        """
        Update indicator with a new data point
        
        Args:
            new_data_point: New data value or dict/Series with named fields
            
        Returns:
            New indicator value
        """
        if not self.is_initialized:
            raise RuntimeError(
                f'Indicator {self.name} must be initialized before updating')
        if hasattr(new_data_point, '__getitem__') and not isinstance(
            new_data_point, (int, float)):
            value = new_data_point[self.input_key]
        else:
            value = new_data_point
        new_value = self._update_incremental(value)
        self.values.append(new_value)
        if len(self.values) > self.window_size:
            self.values = self.values[-self.window_size:]
        return new_value

    def get_value(self) ->float:
        """Get the latest indicator value"""
        if not self.values:
            return None
        return self.values[-1]

    def get_values(self) ->List[float]:
        """Get all available indicator values"""
        return self.values

    def _calculate(self, data) ->np.ndarray:
        """
        Calculate indicator values for a batch of data
        
        Args:
            data: Data in DataFrame or array format
            
        Returns:
            Array of indicator values
        """
        raise NotImplementedError('Subclasses must implement _calculate method'
            )

    def _update_incremental(self, new_value: float) ->float:
        """
        Update indicator state with a new value
        
        Args:
            new_value: New data point
            
        Returns:
            New indicator value
        """
        raise NotImplementedError(
            'Subclasses must implement _update_incremental method')

    def save_state(self) ->Dict[str, Any]:
        """
        Save the current state for later restoration
        
        Returns:
            Dictionary with indicator state
        """
        return {'name': self.name, 'window_size': self.window_size,
            'input_key': self.input_key, 'values': self.values.copy(),
            'is_initialized': self.is_initialized, 'state': self.state.copy()}

    @with_exception_handling
    def restore_state(self, saved_state: Dict[str, Any]) ->bool:
        """
        Restore indicator from a saved state
        
        Args:
            saved_state: State dictionary from save_state()
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.name = saved_state['name']
            self.window_size = saved_state['window_size']
            self.input_key = saved_state['input_key']
            self.values = saved_state['values'].copy()
            self.is_initialized = saved_state['is_initialized']
            self.state = saved_state['state'].copy()
            return True
        except Exception as e:
            self.logger.error(f'Failed to restore state: {str(e)}')
            return False


class IncrementalSMA(IncrementalIndicator):
    """Incrementally calculated Simple Moving Average"""

    @measure_latency
    def _calculate(self, data) ->np.ndarray:
        """Calculate SMA for batch data"""
        if isinstance(data, pd.DataFrame):
            values = data[self.input_key].values
        else:
            values = data
        n = len(values)
        result = np.full(n, np.nan)
        for i in range(n):
            window_start = max(0, i - self.window_size + 1)
            window = values[window_start:i + 1]
            result[i] = np.mean(window)
        window = values[-self.window_size:
            ] if n >= self.window_size else values
        self.state['sum'] = np.sum(window)
        self.state['window'] = deque(window, maxlen=self.window_size)
        self.values = result[~np.isnan(result)].tolist()
        return result

    @measure_latency
    def _update_incremental(self, new_value: float) ->float:
        """Update SMA with a new value"""
        window = self.state['window']
        if len(window) == self.window_size:
            removed = window.popleft()
        else:
            removed = 0
        window.append(new_value)
        self.state['sum'] = self.state['sum'] - removed + new_value
        new_sma = self.state['sum'] / len(window)
        return new_sma


class IncrementalEMA(IncrementalIndicator):
    """Incrementally calculated Exponential Moving Average"""

    def __init__(self, name: str, window_size: int, smoothing: float=2.0,
        input_key: str='close'):
        """
        Initialize EMA with specified parameters
        
        Args:
            name: Indicator name
            window_size: Size of data window (N in formula)
            smoothing: Smoothing factor (typically 2.0)
            input_key: Column name in input data
        """
        super().__init__(name, window_size, input_key)
        self.smoothing = smoothing
        self.alpha = smoothing / (1 + window_size)

    @measure_latency
    def _calculate(self, data) ->np.ndarray:
        """Calculate EMA for batch data"""
        if isinstance(data, pd.DataFrame):
            values = data[self.input_key].values
        else:
            values = data
        n = len(values)
        result = np.full(n, np.nan)
        window_size = min(self.window_size, n)
        result[window_size - 1] = np.mean(values[:window_size])
        for i in range(window_size, n):
            result[i] = values[i] * self.alpha + result[i - 1] * (1 - self.
                alpha)
        self.state['last_ema'] = result[-1]
        self.values = result[~np.isnan(result)].tolist()
        return result

    @measure_latency
    def _update_incremental(self, new_value: float) ->float:
        """Update EMA with a new value"""
        if 'last_ema' not in self.state:
            self.state['last_ema'] = new_value
            return new_value
        last_ema = self.state['last_ema']
        new_ema = new_value * self.alpha + last_ema * (1 - self.alpha)
        self.state['last_ema'] = new_ema
        return new_ema


class IncrementalMACDGenerator:
    """
    Generates MACD (Moving Average Convergence Divergence) indicators incrementally
    using optimized EMA calculations
    """

    def __init__(self, name: str, fast_period: int=12, slow_period: int=26,
        signal_period: int=9, input_key: str='close'):
        """
        Initialize MACD generator
        
        Args:
            name: Indicator name
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            input_key: Column name in input data
        """
        self.name = name
        self.fast_ema = IncrementalEMA(f'{name}_fast', fast_period,
            input_key=input_key)
        self.slow_ema = IncrementalEMA(f'{name}_slow', slow_period,
            input_key=input_key)
        self.signal_ema = IncrementalEMA(f'{name}_signal', signal_period,
            input_key='macd')
        self.macd_values = []
        self.signal_values = []
        self.histogram_values = []
        self.is_initialized = False
        self.logger = logging.getLogger(f'indicator.{name}')

    @measure_latency
    def initialize(self, data: Union[pd.DataFrame, np.ndarray]) ->Dict[str,
        np.ndarray]:
        """
        Initialize MACD with historical data
        
        Args:
            data: Historical data in DataFrame or array format
            
        Returns:
            Dictionary with MACD, Signal, and Histogram values
        """
        fast_values = self.fast_ema.initialize(data)
        slow_values = self.slow_ema.initialize(data)
        macd = fast_values - slow_values
        if isinstance(data, pd.DataFrame):
            signal_data = pd.DataFrame({'macd': macd})
        else:
            signal_data = pd.DataFrame({'macd': macd})
        signal_values = self.signal_ema.initialize(signal_data)
        histogram = macd - signal_values
        self.macd_values = macd[~np.isnan(macd)].tolist()
        self.signal_values = signal_values[~np.isnan(signal_values)].tolist()
        self.histogram_values = histogram[~np.isnan(histogram)].tolist()
        self.is_initialized = True
        return {'macd': macd, 'signal': signal_values, 'histogram': histogram}

    @measure_latency
    def update(self, new_data_point) ->Dict[str, float]:
        """
        Update MACD with a new data point
        
        Args:
            new_data_point: New data value or dict/Series with named fields
            
        Returns:
            Dictionary with new MACD, Signal, and Histogram values
        """
        if not self.is_initialized:
            raise RuntimeError(
                f'MACD {self.name} must be initialized before updating')
        fast_value = self.fast_ema.update(new_data_point)
        slow_value = self.slow_ema.update(new_data_point)
        macd_value = fast_value - slow_value
        self.macd_values.append(macd_value)
        signal_value = self.signal_ema.update({'macd': macd_value})
        self.signal_values.append(signal_value)
        histogram_value = macd_value - signal_value
        self.histogram_values.append(histogram_value)
        return {'macd': macd_value, 'signal': signal_value, 'histogram':
            histogram_value}

    def get_values(self) ->Dict[str, List[float]]:
        """Get all available MACD component values"""
        return {'macd': self.macd_values, 'signal': self.signal_values,
            'histogram': self.histogram_values}

    def save_state(self) ->Dict[str, Any]:
        """
        Save the current state for later restoration
        
        Returns:
            Dictionary with MACD state
        """
        return {'name': self.name, 'fast_ema': self.fast_ema.save_state(),
            'slow_ema': self.slow_ema.save_state(), 'signal_ema': self.
            signal_ema.save_state(), 'macd_values': self.macd_values.copy(),
            'signal_values': self.signal_values.copy(), 'histogram_values':
            self.histogram_values.copy(), 'is_initialized': self.is_initialized
            }

    @with_exception_handling
    def restore_state(self, saved_state: Dict[str, Any]) ->bool:
        """
        Restore MACD from a saved state
        
        Args:
            saved_state: State dictionary from save_state()
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.name = saved_state['name']
            self.fast_ema.restore_state(saved_state['fast_ema'])
            self.slow_ema.restore_state(saved_state['slow_ema'])
            self.signal_ema.restore_state(saved_state['signal_ema'])
            self.macd_values = saved_state['macd_values'].copy()
            self.signal_values = saved_state['signal_values'].copy()
            self.histogram_values = saved_state['histogram_values'].copy()
            self.is_initialized = saved_state['is_initialized']
            return True
        except Exception as e:
            self.logger.error(f'Failed to restore state: {str(e)}')
            return False


class IncrementalRSI(IncrementalIndicator):
    """Incrementally calculated Relative Strength Index"""

    def __init__(self, name: str, window_size: int=14, input_key: str='close'):
        """
        Initialize RSI calculator
        
        Args:
            name: Indicator name
            window_size: Size of data window
            input_key: Column name in input data
        """
        super().__init__(name, window_size, input_key)

    @measure_latency
    def _calculate(self, data) ->np.ndarray:
        """Calculate RSI for batch data"""
        if isinstance(data, pd.DataFrame):
            values = data[self.input_key].values
        else:
            values = data
        n = len(values)
        result = np.full(n, np.nan)
        if n < 2:
            return result
        changes = np.diff(values)
        for i in range(self.window_size, n):
            window = changes[i - self.window_size:i]
            gains = window[window > 0]
            losses = -window[window < 0]
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            if avg_loss == 0:
                result[i] = 100
            else:
                rs = avg_gain / avg_loss
                result[i] = 100 - 100 / (1 + rs)
        last_changes = changes[-self.window_size:]
        last_gains = last_changes[last_changes > 0]
        last_losses = -last_changes[last_changes < 0]
        self.state['last_avg_gain'] = np.mean(last_gains) if len(last_gains
            ) > 0 else 0
        self.state['last_avg_loss'] = np.mean(last_losses) if len(last_losses
            ) > 0 else 0
        self.state['last_price'] = values[-1]
        self.values = result[~np.isnan(result)].tolist()
        return result

    @measure_latency
    def _update_incremental(self, new_value: float) ->float:
        """Update RSI with a new value"""
        if 'last_price' not in self.state:
            self.state['last_price'] = new_value
            self.state['last_avg_gain'] = 0
            self.state['last_avg_loss'] = 0
            return 50
        change = new_value - self.state['last_price']
        self.state['last_price'] = new_value
        alpha = 1 / self.window_size
        if change > 0:
            avg_gain = (self.state['last_avg_gain'] * (self.window_size - 1
                ) + change) / self.window_size
            avg_loss = self.state['last_avg_loss'] * (self.window_size - 1
                ) / self.window_size
        else:
            avg_gain = self.state['last_avg_gain'] * (self.window_size - 1
                ) / self.window_size
            avg_loss = (self.state['last_avg_loss'] * (self.window_size - 1
                ) + abs(change)) / self.window_size
        self.state['last_avg_gain'] = avg_gain
        self.state['last_avg_loss'] = avg_loss
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - 100 / (1 + rs)
        return rsi


class ParallelIndicatorCalculator:
    """
    Calculates multiple indicators in parallel using a thread pool
    for improved performance with large datasets or many indicators
    """

    def __init__(self, max_workers: int=4):
        """
        Initialize parallel calculator
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)

    @measure_latency
    @with_exception_handling
    def calculate_indicators(self, data: pd.DataFrame, indicators: List[
        IncrementalIndicator]) ->Dict[str, np.ndarray]:
        """
        Calculate multiple indicators in parallel
        
        Args:
            data: DataFrame with input data
            indicators: List of indicator objects to calculate
            
        Returns:
            Dictionary mapping indicator names to result arrays
        """
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for indicator in indicators:
                future = executor.submit(indicator.initialize, data)
                futures[future] = indicator.name
            for future in as_completed(futures):
                indicator_name = futures[future]
                try:
                    result = future.result()
                    results[indicator_name] = result
                except Exception as e:
                    self.logger.error(
                        f'Error calculating {indicator_name}: {str(e)}')
        return results


class IndicatorManager:
    """
    Manages the creation, calculation, and persistence of technical indicators
    with optimized incremental updates and state management
    """

    def __init__(self, state_persistence_path: Optional[str]=None):
        """
        Initialize indicator manager
        
        Args:
            state_persistence_path: Optional path for saving/loading indicator states
        """
        self.indicators = {}
        self.state_persistence_path = state_persistence_path
        self.parallel_calculator = ParallelIndicatorCalculator()
        self.logger = logging.getLogger(__name__)

    def create_indicator(self, indicator_type: str, name: str, **kwargs
        ) ->Union[IncrementalIndicator, IncrementalMACDGenerator]:
        """
        Create a new indicator of specified type
        
        Args:
            indicator_type: Type of indicator ('sma', 'ema', 'rsi', 'macd')
            name: Unique name for the indicator
            **kwargs: Parameters for the indicator
            
        Returns:
            Created indicator object
        """
        if name in self.indicators:
            self.logger.warning(
                f'Indicator {name} already exists, returning existing instance'
                )
            return self.indicators[name]
        indicator = None
        if indicator_type.lower() == 'sma':
            indicator = IncrementalSMA(name, **kwargs)
        elif indicator_type.lower() == 'ema':
            indicator = IncrementalEMA(name, **kwargs)
        elif indicator_type.lower() == 'rsi':
            indicator = IncrementalRSI(name, **kwargs)
        elif indicator_type.lower() == 'macd':
            indicator = IncrementalMACDGenerator(name, **kwargs)
        else:
            raise ValueError(f'Unknown indicator type: {indicator_type}')
        self.indicators[name] = indicator
        return indicator

    @with_exception_handling
    def calculate_all(self, data: pd.DataFrame, use_parallel: bool=True
        ) ->Dict[str, np.ndarray]:
        """
        Calculate all registered indicators with the provided data
        
        Args:
            data: DataFrame with input data
            use_parallel: Whether to use parallel calculation
            
        Returns:
            Dictionary mapping indicator names to result arrays
        """
        results = {}
        if use_parallel and len(self.indicators) > 1:
            regular_indicators = [ind for ind in self.indicators.values() if
                not isinstance(ind, IncrementalMACDGenerator)]
            macd_indicators = [ind for ind in self.indicators.values() if
                isinstance(ind, IncrementalMACDGenerator)]
            if regular_indicators:
                regular_results = (self.parallel_calculator.
                    calculate_indicators(data, regular_indicators))
                results.update(regular_results)
            for indicator in macd_indicators:
                macd_results = indicator.initialize(data)
                results[indicator.name] = macd_results
        else:
            for name, indicator in self.indicators.items():
                try:
                    if isinstance(indicator, IncrementalMACDGenerator):
                        results[name] = indicator.initialize(data)
                    else:
                        results[name] = indicator.initialize(data)
                except Exception as e:
                    self.logger.error(f'Error calculating {name}: {str(e)}')
        return results

    @with_exception_handling
    def update_all(self, new_data_point: Union[Dict[str, float], pd.Series]
        ) ->Dict[str, Union[float, Dict[str, float]]]:
        """
        Update all indicators with a new data point
        
        Args:
            new_data_point: New data point as dict or Series
            
        Returns:
            Dictionary mapping indicator names to updated values
        """
        results = {}
        for name, indicator in self.indicators.items():
            try:
                if isinstance(indicator, IncrementalMACDGenerator):
                    results[name] = indicator.update(new_data_point)
                else:
                    results[name] = indicator.update(new_data_point)
            except Exception as e:
                self.logger.error(f'Error updating {name}: {str(e)}')
                results[name] = None
        return results

    def get_indicator(self, name: str) ->Optional[Union[
        IncrementalIndicator, IncrementalMACDGenerator]]:
        """Get an indicator by name"""
        return self.indicators.get(name)

    def remove_indicator(self, name: str) ->bool:
        """Remove an indicator by name"""
        if name in self.indicators:
            del self.indicators[name]
            return True
        return False

    @measure_latency
    @with_exception_handling
    def save_all_states(self) ->bool:
        """
        Save states of all indicators to disk
        
        Returns:
            True if successful, False otherwise
        """
        if not self.state_persistence_path:
            self.logger.warning(
                'No persistence path specified, cannot save states')
            return False
        try:
            import json
            import os
            os.makedirs(os.path.dirname(self.state_persistence_path),
                exist_ok=True)
            states = {}
            for name, indicator in self.indicators.items():
                states[name] = {'type': type(indicator).__name__, 'state':
                    indicator.save_state()}
            with open(self.state_persistence_path, 'w') as f:
                json.dump(states, f)
            self.logger.info(
                f'Saved {len(states)} indicator states to {self.state_persistence_path}'
                )
            return True
        except Exception as e:
            self.logger.error(f'Failed to save indicator states: {str(e)}')
            return False

    @measure_latency
    @with_exception_handling
    def load_all_states(self) ->bool:
        """
        Load states of all indicators from disk
        
        Returns:
            True if successful, False otherwise
        """
        if not self.state_persistence_path:
            self.logger.warning(
                'No persistence path specified, cannot load states')
            return False
        try:
            import json
            import os
            if not os.path.exists(self.state_persistence_path):
                self.logger.warning(
                    f'State file {self.state_persistence_path} does not exist')
                return False
            with open(self.state_persistence_path, 'r') as f:
                states = json.load(f)
            for name, data in states.items():
                indicator_type = data['type']
                state = data['state']
                if indicator_type == 'IncrementalSMA':
                    indicator = IncrementalSMA(name, state['window_size'],
                        state['input_key'])
                elif indicator_type == 'IncrementalEMA':
                    indicator = IncrementalEMA(name, state['window_size'],
                        input_key=state['input_key'])
                elif indicator_type == 'IncrementalRSI':
                    indicator = IncrementalRSI(name, state['window_size'],
                        state['input_key'])
                elif indicator_type == 'IncrementalMACDGenerator':
                    fast_period = state['fast_ema']['window_size']
                    slow_period = state['slow_ema']['window_size']
                    signal_period = state['signal_ema']['window_size']
                    input_key = state['fast_ema']['input_key']
                    indicator = IncrementalMACDGenerator(name, fast_period,
                        slow_period, signal_period, input_key)
                else:
                    self.logger.warning(
                        f'Unknown indicator type: {indicator_type}')
                    continue
                indicator.restore_state(state)
                self.indicators[name] = indicator
            self.logger.info(
                f'Loaded {len(states)} indicator states from {self.state_persistence_path}'
                )
            return True
        except Exception as e:
            self.logger.error(f'Failed to load indicator states: {str(e)}')
            return False

    def get_performance_metrics(self) ->Dict[str, Any]:
        """Get performance metrics for all indicator calculations"""
        return performance_monitor.get_all_metrics()
