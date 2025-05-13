"""
Performance Optimization Metrics Exporter.

This module exports metrics from the performance optimization system to the monitoring service.
It tracks GPU utilization, calculation speedup, memory optimization, and other performance metrics.
"""
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import json
import os
from prometheus_client import Gauge, Histogram, Counter, Summary
import numpy as np
import psutil
import threading
logger = logging.getLogger(__name__)
CALCULATION_LATENCY = Histogram('indicator_calculation_latency_seconds',
    'Indicator calculation latency in seconds', ['indicator_name',
    'enhanced', 'data_size'])
MEMORY_USAGE = Gauge('indicator_memory_usage_mb',
    'Indicator memory usage in MB', ['indicator_name', 'enhanced', 'data_size']
    )
GPU_UTILIZATION = Gauge('gpu_utilization_percent',
    'GPU utilization percentage', ['gpu_id'])
CACHE_HIT_RATIO = Gauge('indicator_cache_hit_ratio',
    'Cache hit ratio for indicator calculations', ['indicator_name'])
CALCULATION_COUNT = Counter('indicator_calculation_count',
    'Number of indicator calculations performed', ['indicator_name',
    'enhanced', 'succeeded'])
SPEEDUP_FACTOR = Gauge('indicator_speedup_factor',
    'Speedup factor compared to non-enhanced calculation', [
    'indicator_name', 'data_size'])
MEMORY_REDUCTION_FACTOR = Gauge('indicator_memory_reduction_factor',
    'Memory reduction factor compared to non-enhanced calculation', [
    'indicator_name', 'data_size'])


from monitoring_alerting_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class PerformanceOptimizationExporter:
    """
    Exports performance metrics from the optimization system to monitoring service.
    """

    def __init__(self, metrics_dir: Optional[str]=None, export_interval: int=60
        ):
        """
        Initialize the performance metrics exporter.
        
        Args:
            metrics_dir: Directory to store metrics files (None to disable file export)
            export_interval: Interval in seconds for exporting metrics
        """
        self.metrics_dir = metrics_dir
        if metrics_dir and not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir, exist_ok=True)
        self.export_interval = export_interval
        self.performance_data = {}
        self.benchmark_results = {}
        self.has_gpu = self._check_gpu_available()
        self.gpu_monitor_thread = None
        self.stop_monitor = threading.Event()
        if self.has_gpu:
            self._start_gpu_monitor()

    def _start_gpu_monitor(self):
        """Start GPU utilization monitoring thread."""
        self.stop_monitor.clear()
        self.gpu_monitor_thread = threading.Thread(target=self._monitor_gpu,
            daemon=True)
        self.gpu_monitor_thread.start()
        logger.info('Started GPU monitoring')

    def _stop_gpu_monitor(self):
        """Stop GPU utilization monitoring thread."""
        if self.gpu_monitor_thread and self.gpu_monitor_thread.is_alive():
            self.stop_monitor.set()
            self.gpu_monitor_thread.join(timeout=1)
            logger.info('Stopped GPU monitoring')

    @with_exception_handling
    def _check_gpu_available(self) ->bool:
        """Check if GPU monitoring is available."""
        try:
            import torch
            if torch.cuda.is_available():
                return True
        except ImportError:
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    return True
            except ImportError:
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi'], stdout=
                        subprocess.PIPE, stderr=subprocess.PIPE)
                    return result.returncode == 0
                except:
                    pass
        return False

    @with_exception_handling
    def _monitor_gpu(self):
        """Monitor GPU utilization and export metrics."""
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False
        try:
            import py3nvml.py3nvml as nvml
            has_nvml = True
            nvml.nvmlInit()
        except ImportError:
            has_nvml = False
        while not self.stop_monitor.wait(5.0):
            try:
                if has_torch and torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        memory_allocated = torch.cuda.memory_allocated(i)
                        memory_reserved = torch.cuda.memory_reserved(i)
                        total_memory = torch.cuda.get_device_properties(i
                            ).total_memory
                        utilization = memory_allocated / total_memory * 100
                        GPU_UTILIZATION.labels(gpu_id=f'cuda:{i}').set(
                            utilization)
                elif has_nvml:
                    device_count = nvml.nvmlDeviceGetCount()
                    for i in range(device_count):
                        handle = nvml.nvmlDeviceGetHandleByIndex(i)
                        util = nvml.nvmlDeviceGetUtilizationRates(handle)
                        GPU_UTILIZATION.labels(gpu_id=f'gpu:{i}').set(util.gpu)
            except Exception as e:
                logger.error(f'Error monitoring GPU: {e}')

    def record_calculation_metrics(self, indicator_name: str, enhanced:
        bool, data_size: int, execution_time: float, memory_usage: Optional
        [float]=None, succeeded: bool=True) ->None:
        """
        Record metrics for a single indicator calculation.
        
        Args:
            indicator_name: Name of the indicator
            enhanced: Whether using performance enhancement
            data_size: Size of input data (number of rows)
            execution_time: Execution time in seconds
            memory_usage: Memory usage in MB (None if not measured)
            succeeded: Whether calculation succeeded
        """
        key = f'{indicator_name}_{data_size}'
        if key not in self.performance_data:
            self.performance_data[key] = {'standard': {'times': [],
                'memory': []}, 'enhanced': {'times': [], 'memory': []}}
        category = 'enhanced' if enhanced else 'standard'
        self.performance_data[key][category]['times'].append(execution_time)
        if memory_usage is not None:
            self.performance_data[key][category]['memory'].append(memory_usage)
        CALCULATION_LATENCY.labels(indicator_name=indicator_name, enhanced=
            str(enhanced), data_size=str(data_size)).observe(execution_time)
        if memory_usage is not None:
            MEMORY_USAGE.labels(indicator_name=indicator_name, enhanced=str
                (enhanced), data_size=str(data_size)).set(memory_usage)
        CALCULATION_COUNT.labels(indicator_name=indicator_name, enhanced=
            str(enhanced), succeeded=str(succeeded)).inc()
        self._update_performance_ratios(indicator_name, data_size)

    def record_cache_metrics(self, indicator_name: str, hits: int, misses: int
        ) ->None:
        """
        Record cache hit metrics for indicator calculations.
        
        Args:
            indicator_name: Name of the indicator
            hits: Number of cache hits
            misses: Number of cache misses
        """
        total = hits + misses
        hit_ratio = hits / total if total > 0 else 0
        CACHE_HIT_RATIO.labels(indicator_name=indicator_name).set(hit_ratio)

    def _update_performance_ratios(self, indicator_name: str, data_size: int
        ) ->None:
        """Update speedup and memory reduction metrics if both standard and enhanced data exists."""
        key = f'{indicator_name}_{data_size}'
        if key not in self.performance_data:
            return
        data = self.performance_data[key]
        if data['standard']['times'] and data['enhanced']['times']:
            std_time = np.median(data['standard']['times'])
            enh_time = np.median(data['enhanced']['times'])
            if enh_time > 0:
                speedup = std_time / enh_time
                SPEEDUP_FACTOR.labels(indicator_name=indicator_name,
                    data_size=str(data_size)).set(speedup)
        if data['standard']['memory'] and data['enhanced']['memory']:
            std_mem = np.median(data['standard']['memory'])
            enh_mem = np.median(data['enhanced']['memory'])
            if enh_mem > 0:
                reduction = std_mem / enh_mem
                MEMORY_REDUCTION_FACTOR.labels(indicator_name=
                    indicator_name, data_size=str(data_size)).set(reduction)

    def record_benchmark_result(self, benchmark_result: Dict[str, Any]) ->None:
        """
        Record a complete benchmark result.
        
        Args:
            benchmark_result: Benchmark result dictionary 
                (with name, standard_time, enhanced_time, speedup, etc.)
        """
        name = benchmark_result.get('name', 'unknown')
        self.benchmark_results[name] = benchmark_result
        standard_time = benchmark_result.get('standard_time')
        enhanced_time = benchmark_result.get('enhanced_time')
        speedup = benchmark_result.get('speedup')
        if speedup is not None:
            parts = name.split('Size_')
            data_size = parts[1] if len(parts) > 1 else 'unknown'
            SPEEDUP_FACTOR.labels(indicator_name=name.split('_')[0],
                data_size=data_size).set(speedup)
        if standard_time is not None:
            data_size = 'unknown'
            parts = name.split('Size_')
            if len(parts) > 1:
                data_size = parts[1]
                indicator_name = parts[0].strip('_')
            else:
                indicator_name = name
            self.record_calculation_metrics(indicator_name=indicator_name,
                enhanced=False, data_size=data_size, execution_time=
                standard_time, memory_usage=benchmark_result.get(
                'standard_memory'))
            if enhanced_time is not None:
                self.record_calculation_metrics(indicator_name=
                    indicator_name, enhanced=True, data_size=data_size,
                    execution_time=enhanced_time, memory_usage=
                    benchmark_result.get('enhanced_memory'))

    @with_exception_handling
    def export_metrics_to_file(self) ->None:
        """Export current metrics to a JSON file."""
        if not self.metrics_dir:
            return
        try:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            perf_filename = os.path.join(self.metrics_dir,
                f'performance_metrics_{timestamp}.json')
            with open(perf_filename, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
            if self.benchmark_results:
                bench_filename = os.path.join(self.metrics_dir,
                    f'benchmark_results_{timestamp}.json')
                with open(bench_filename, 'w') as f:
                    json.dump(self.benchmark_results, f, indent=2)
            logger.info(f'Exported metrics to {self.metrics_dir}')
        except Exception as e:
            logger.error(f'Error exporting metrics to file: {e}')

    def start_periodic_export(self) ->None:
        """Start periodic export of metrics to files."""

        def _export_job():
    """
     export job.
    
    """

            while True:
                time.sleep(self.export_interval)
                self.export_metrics_to_file()
        thread = threading.Thread(target=_export_job, daemon=True)
        thread.start()

    def __del__(self):
        """Clean up resources."""
        self._stop_gpu_monitor()


_metrics_exporter = None


def get_metrics_exporter() ->PerformanceOptimizationExporter:
    """Get the global metrics exporter instance."""
    global _metrics_exporter
    if _metrics_exporter is None:
        metrics_dir = os.environ.get('PERFORMANCE_METRICS_DIR', os.path.
            join(os.path.dirname(os.path.dirname(__file__)), 'metrics',
            'performance'))
        _metrics_exporter = PerformanceOptimizationExporter(metrics_dir=
            metrics_dir)
    return _metrics_exporter
