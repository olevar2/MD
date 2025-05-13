"""
Performance Profiling Module for Strategy Execution Pipeline

This module provides tools to measure and analyze the performance of the strategy execution pipeline,
identify bottlenecks, and capture execution metrics for optimization purposes.
"""
import time
import logging
import functools
import statistics
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
import json
import os
from pathlib import Path
import threading
import traceback
logger = logging.getLogger(__name__)


from strategy_execution_engine.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ExecutionProfiler:
    """
    Performance profiler for strategy execution pipeline components.
    Tracks execution times, identifies bottlenecks, and provides optimization recommendations.
    """

    def __init__(self, enabled: bool=True, output_dir: Optional[str]=None):
        """
        Initialize the execution profiler.
        
        Args:
            enabled: Whether profiling is enabled
            output_dir: Optional directory to save profiling results
        """
        self.enabled = enabled
        self.output_dir = output_dir or 'profiling_results'
        self._ensure_output_dir()
        self.execution_times = {}
        self.call_counts = {}
        self.memory_usage = {}
        self.bottlenecks = {}
        self._active_spans = {}
        self._lock = threading.RLock()
        self.bottleneck_thresholds = {'critical': 1.0, 'warning': 0.5,
            'notice': 0.1}

    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        if self.output_dir:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def start_span(self, component: str, operation: str, metadata: Optional
        [Dict[str, Any]]=None) ->str:
        """
        Start timing an operation.
        
        Args:
            component: The component being timed (e.g., 'SignalAggregator')
            operation: The operation being performed (e.g., 'process_signals')
            metadata: Additional context about the operation
            
        Returns:
            A span ID that can be used to stop the timing
        """
        if not self.enabled:
            return f'{component}_{operation}_{datetime.now().timestamp()}'
        with self._lock:
            span_id = f'{component}_{operation}_{datetime.now().timestamp()}'
            self._active_spans[span_id] = {'component': component,
                'operation': operation, 'start_time': time.time(),
                'metadata': metadata or {}}
            return span_id

    def stop_span(self, span_id: str) ->Optional[float]:
        """
        Stop timing an operation and record its duration.
        
        Args:
            span_id: The span ID returned from start_span
            
        Returns:
            The duration of the operation in seconds, or None if profiling is disabled
        """
        if not self.enabled:
            return None
        with self._lock:
            if span_id not in self._active_spans:
                logger.warning(f'Attempt to stop unknown span: {span_id}')
                return None
            span_data = self._active_spans.pop(span_id)
            duration = time.time() - span_data['start_time']
            component = span_data['component']
            operation = span_data['operation']
            key = f'{component}.{operation}'
            if key not in self.execution_times:
                self.execution_times[key] = []
            self.execution_times[key].append(duration)
            self.call_counts[key] = self.call_counts.get(key, 0) + 1
            self._check_bottleneck(key, duration, span_data)
            return duration

    def _check_bottleneck(self, key: str, duration: float, span_data: Dict[
        str, Any]):
        """
        Check if the operation duration indicates a bottleneck.
        
        Args:
            key: The component.operation key
            duration: The operation duration in seconds
            span_data: The span data including metadata
        """
        severity = None
        for level, threshold in self.bottleneck_thresholds.items():
            if duration > threshold:
                severity = level
                break
        if severity:
            if key not in self.bottlenecks:
                self.bottlenecks[key] = {'count': 0, 'total_time': 0,
                    'max_time': 0, 'instances': []}
            bottleneck = self.bottlenecks[key]
            bottleneck['count'] += 1
            bottleneck['total_time'] += duration
            bottleneck['max_time'] = max(bottleneck['max_time'], duration)
            if len(bottleneck['instances']) < 100:
                instance = {'duration': duration, 'timestamp': datetime.now
                    ().isoformat(), 'severity': severity, 'metadata':
                    span_data.get('metadata', {})}
                bottleneck['instances'].append(instance)

    @with_exception_handling
    def profile(self, component: str, operation: str):
        """
        Decorator to profile a function.
        
        Args:
            component: The component name
            operation: The operation name
            
        Returns:
            Decorated function
        """

        @with_exception_handling
        def decorator(func):
    """
    Decorator.
    
    Args:
        func: Description of func
    
    """


            @functools.wraps(func)
            @with_exception_handling
            def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

                if not self.enabled:
                    return func(*args, **kwargs)
                metadata = {'args_length': len(args), 'kwargs_keys': list(
                    kwargs.keys())}
                span_id = self.start_span(component, operation, metadata)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.stop_span(span_id)
            return wrapper
        return decorator

    def get_summary(self) ->Dict[str, Any]:
        """
        Get a summary of profiling results.
        
        Returns:
            Dictionary with profiling summary
        """
        if not self.enabled:
            return {'enabled': False}
        summary = {'enabled': True, 'total_operations': sum(self.
            call_counts.values()), 'unique_operations': len(self.
            call_counts), 'bottlenecks': len(self.bottlenecks),
            'operation_stats': {}, 'top_bottlenecks': []}
        for key, times in self.execution_times.items():
            if not times:
                continue
            stats = {'count': len(times), 'total_time': sum(times),
                'avg_time': statistics.mean(times), 'min_time': min(times),
                'max_time': max(times), 'p95_time': sorted(times)[int(len(
                times) * 0.95)] if len(times) > 20 else max(times)}
            summary['operation_stats'][key] = stats
        if self.bottlenecks:
            bottleneck_list = [{'operation': key, 'count': data['count'],
                'total_time': data['total_time'], 'max_time': data[
                'max_time'], 'avg_time': data['total_time'] / data['count'] if
                data['count'] > 0 else 0} for key, data in self.bottlenecks
                .items()]
            bottleneck_list.sort(key=lambda x: x['total_time'], reverse=True)
            summary['top_bottlenecks'] = bottleneck_list[:10]
        return summary

    def save_results(self, filename: Optional[str]=None) ->str:
        """
        Save profiling results to file.
        
        Args:
            filename: Optional filename, default is auto-generated
            
        Returns:
            Path to the saved file
        """
        if not self.enabled:
            return ''
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'profile_results_{timestamp}.json'
        filepath = os.path.join(self.output_dir, filename)
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        return filepath

    def reset(self):
        """Reset all profiling data."""
        with self._lock:
            self.execution_times = {}
            self.call_counts = {}
            self.memory_usage = {}
            self.bottlenecks = {}
            self._active_spans = {}


global_profiler = ExecutionProfiler(enabled=True)


def profile_execution(component: str, operation: str):
    """
    Decorator to profile a function using the global profiler.
    
    Args:
        component: The component name
        operation: The operation name
        
    Returns:
        Decorated function
    """
    return global_profiler.profile(component, operation)


class BatchProcessingOptimizer:
    """
    Utility for optimizing batch processing in the strategy execution pipeline.
    """

    def __init__(self, min_batch_size: int=5, max_batch_size: int=100):
        """
        Initialize the batch processing optimizer.
        
        Args:
            min_batch_size: Minimum batch size to enable batching
            max_batch_size: Maximum batch size for processing
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.batch_size_history = {}

    def get_optimal_batch_size(self, operation: str, items_count: int) ->int:
        """
        Calculate the optimal batch size for a specific operation.
        
        Args:
            operation: The operation identifier
            items_count: The number of items to process
            
        Returns:
            The recommended batch size
        """
        if items_count < self.min_batch_size:
            return items_count
        if operation in self.batch_size_history:
            return min(self.batch_size_history[operation], self.
                max_batch_size, items_count)
        default_size = min(max(items_count // 4, self.min_batch_size), self
            .max_batch_size)
        self.batch_size_history[operation] = default_size
        return default_size

    def update_performance_data(self, operation: str, batch_size: int,
        duration: float):
        """
        Update batch size optimization based on performance data.
        
        Args:
            operation: The operation identifier
            batch_size: The batch size used
            duration: The execution duration per item
        """
        pass


profiler = global_profiler
batch_optimizer = BatchProcessingOptimizer()
""""""
