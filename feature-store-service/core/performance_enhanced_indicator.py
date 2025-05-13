"""
Performance Enhanced Indicator Module.

This module provides an extension to BaseIndicator with performance optimizations
using GPU acceleration, advanced calculation, load balancing, and memory optimization.
"""
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import time
import logging
import os
from functools import wraps
from core.base_indicator import BaseIndicator
from core.gpu_acceleration import GPUAccelerator, is_gpu_available
from core.advanced_calculation import smart_cache, pre_aggregator, lazy_calculator, incremental_calculator
from core.load_balancing import get_load_balancer, ComputationPriority
from services.memory_optimization import get_historical_data_manager, MemoryOptimizer, DataPrecision
logger = logging.getLogger(__name__)
try:
    from monitoring_alerting_service.metrics_exporters.performance_optimization_exporter import get_metrics_exporter
    has_metrics_exporter = True
except ImportError:
    has_metrics_exporter = False
    logger.debug(
        'Performance metrics exporter not available. Metrics will only be logged locally.'
        )


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@with_exception_handling
def performance_monitored(func):
    """
    Decorator to monitor performance of indicator calculations.
    
    Args:
        func: Function to monitor
        
    Returns:
        Decorated function with performance monitoring
    """

    @wraps(func)
    @with_exception_handling
    def wrapper(self, *args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

        start_time = time.time()
        try:
            import psutil
            initial_memory = psutil.Process().memory_info().rss
        except:
            initial_memory = None
        result = func(self, *args, **kwargs)
        elapsed_time = time.time() - start_time
        memory_delta_mb = None
        if initial_memory is not None:
            try:
                current_memory = psutil.Process().memory_info().rss
                memory_delta = current_memory - initial_memory
                memory_delta_mb = memory_delta / 1024 / 1024
            except:
                memory_delta = None
        indicator_name = self.__class__.__name__ if hasattr(self, '__class__'
            ) else 'Unknown'
        log_message = (
            f'Performance: {indicator_name}.{func.__name__} completed in {elapsed_time:.4f}s'
            )
        if memory_delta_mb is not None:
            log_message += f' (memory delta: {memory_delta_mb:.2f} MB)'
        logger.info(log_message)
        if hasattr(self, '_performance_metrics'):
            metric_data = {'indicator': indicator_name, 'function': func.
                __name__, 'elapsed_time': elapsed_time, 'memory_delta_mb':
                memory_delta_mb, 'timestamp': time.time()}
            data_size = None
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    data_size = len(arg)
                    metric_data['data_size'] = data_size
                    break
            self._performance_metrics.append(metric_data)
            if has_metrics_exporter and func.__name__ == 'calculate':
                try:
                    exporter = get_metrics_exporter()
                    is_enhanced = hasattr(self, '_use_gpu') or hasattr(self,
                        '_use_advanced_calc')
                    exporter.record_calculation_metrics(indicator_name=
                        indicator_name, enhanced=is_enhanced, data_size=
                        data_size or 0, execution_time=elapsed_time,
                        memory_usage=memory_delta_mb, succeeded=True)
                except Exception as e:
                    logger.warning(f'Failed to export performance metrics: {e}'
                        )
        return result
    return wrapper


class PerformanceEnhancedIndicator(BaseIndicator):
    """
    Base class for indicators with performance optimizations.
    
    This class extends BaseIndicator with performance enhancements including
    GPU acceleration, advanced calculation techniques, load balancing,
    and memory optimization.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a performance-enhanced indicator.
        
        Args:
            *args: Positional arguments for BaseIndicator
            **kwargs: Keyword arguments for BaseIndicator and performance options
        """
        self._use_gpu = kwargs.pop('use_gpu', True)
        self._use_advanced_calc = kwargs.pop('use_advanced_calc', True)
        self._use_load_balancing = kwargs.pop('use_load_balancing', True)
        self._use_memory_optimization = kwargs.pop('use_memory_optimization',
            True)
        self._computation_priority = kwargs.pop('computation_priority',
            ComputationPriority.NORMAL)
        self._performance_metrics = []
        super().__init__(*args, **kwargs)
        if self._use_gpu and is_gpu_available():
            self._gpu_accelerator = GPUAccelerator()
        else:
            self._gpu_accelerator = None
        if self._use_memory_optimization:
            self._data_manager = get_historical_data_manager()
        self._cache_key = f'{self.__class__.__name__}_{id(self)}'

    @performance_monitored
    def calculate(self, data: pd.DataFrame, *args, **kwargs) ->pd.DataFrame:
        """
        Calculate the indicator with performance optimizations.
        
        This method wraps the _calculate implementation with performance enhancements
        like GPU acceleration, caching, and memory optimization.
        
        Args:
            data: Input DataFrame
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            DataFrame with indicator values
        """
        if self._use_memory_optimization and len(data) > 10000:
            data = MemoryOptimizer.optimize_dataframe(data)
        if self._use_advanced_calc:
            cache_hit, cached_result = smart_cache.get(
                f'{self._cache_key}.calculate', (data.shape, data.index[0],
                data.index[-1]), kwargs)
            if cache_hit:
                return cached_result
        if self._use_load_balancing and len(data) > 50000:
            load_balancer = get_load_balancer()
            task_id = load_balancer.submit_task(func=self._calculate_impl,
                args=(data, *args), kwargs=kwargs, priority=self.
                _computation_priority, cpu_intensive=True)
            task_result = load_balancer.get_task_result(task_id, wait=True)
            result = task_result['result']
        else:
            result = self._calculate_impl(data, *args, **kwargs)
        if self._use_advanced_calc:
            smart_cache.set(f'{self._cache_key}.calculate', (data.shape,
                data.index[0], data.index[-1]), kwargs, result)
        return result

    def _calculate_impl(self, data: pd.DataFrame, *args, **kwargs
        ) ->pd.DataFrame:
        """
        Implementation of the indicator calculation.
        
        This method should be overridden by concrete indicator implementations.
        It will be called by the calculate() method with performance enhancements.
        
        Args:
            data: Input DataFrame
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            DataFrame with indicator values
        """
        raise NotImplementedError('Subclasses must implement _calculate_impl')

    def get_performance_metrics(self) ->List[Dict[str, Any]]:
        """
        Get performance metrics for this indicator.
        
        Returns:
            List of performance metric dictionaries
        """
        return self._performance_metrics

    def clear_performance_metrics(self) ->None:
        """
        Clear collected performance metrics.
        """
        self._performance_metrics = []


@with_exception_handling
def gpu_accelerated_operation(operation_name: str, func, *args, **kwargs):
    """
    Execute an operation with GPU acceleration if available.
    
    Args:
        operation_name: Name of the operation (for logging)
        func: Operation function to execute
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result from the function
    """
    start_time = time.time()
    if is_gpu_available():
        accelerator = GPUAccelerator()
        gpu_args = [(accelerator.to_gpu(arg) if isinstance(arg, (np.ndarray,
            pd.DataFrame, pd.Series)) else arg) for arg in args]
        try:
            result = func(*gpu_args, **kwargs)
            if hasattr(result, 'cpu') or hasattr(result, 'numpy'):
                result = accelerator.to_cpu(result)
            elapsed = time.time() - start_time
            logger.debug(
                f'GPU operation {operation_name} completed in {elapsed:.4f}s')
            return result
        except Exception as e:
            logger.warning(
                f'GPU operation {operation_name} failed: {e}. Falling back to CPU.'
                )
    result = func(*args, **kwargs)
    elapsed = time.time() - start_time
    logger.debug(f'CPU operation {operation_name} completed in {elapsed:.4f}s')
    return result
