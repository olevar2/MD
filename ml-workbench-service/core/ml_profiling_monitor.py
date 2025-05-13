"""
ML Profiling and Monitoring Integration

This module provides tools for profiling ML models and integrating with
monitoring systems to track performance metrics.

It includes:
- ML model profiling (CPU, memory, latency)
- Distributed tracing integration
- Prometheus metrics integration
- Grafana dashboard generation
- Alerting configuration
"""
import logging
import time
import os
import json
import uuid
import subprocess
import threading
import socket
import re
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import psutil
import tempfile
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
logger = logging.getLogger(__name__)


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class MLProfilingMonitor:
    """
    Provides profiling and monitoring integration for ML models.

    This class provides methods for:
    - Profiling ML models
    - Integrating with monitoring systems
    - Generating dashboards
    - Configuring alerts
    """

    def __init__(self, model_path: Optional[str]=None, model_object:
        Optional[Any]=None, framework: str='tensorflow', output_dir: str=
        './ml_profiling', model_name: str='model', prometheus_port: int=
        8000, prometheus_endpoint: str='/metrics'):
        """
        Initialize the ML profiling monitor.

        Args:
            model_path: Path to the model file
            model_object: Model object (alternative to model_path)
            framework: ML framework the model belongs to
            output_dir: Directory for profiling outputs
            model_name: Name of the model
            prometheus_port: Port for Prometheus metrics
            prometheus_endpoint: Endpoint for Prometheus metrics
        """
        self.model_path = model_path
        self.model_object = model_object
        self.framework = framework.lower()
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.prometheus_port = prometheus_port
        self.prometheus_endpoint = prometheus_endpoint
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self._validate_framework()
        if model_path and not model_object:
            self.model_object = self._load_model(model_path)
        self.profiling_results = {}
        self.monitoring_server = None
        self.monitoring_thread = None
        self.is_monitoring = False
        self.metrics = {'inference_count': 0, 'inference_latency_ms': [],
            'cpu_usage_percent': [], 'memory_usage_mb': [], 'error_count': 
            0, 'batch_sizes': [], 'timestamp': []}

    def _validate_framework(self):
        """Validate that the requested framework is available."""
        if self.framework == 'tensorflow' and not TENSORFLOW_AVAILABLE:
            raise ImportError(
                'TensorFlow is not available. Please install it to use this framework.'
                )
        elif self.framework == 'pytorch' and not PYTORCH_AVAILABLE:
            raise ImportError(
                'PyTorch is not available. Please install it to use this framework.'
                )
        elif self.framework == 'onnx' and not ONNX_AVAILABLE:
            raise ImportError(
                'ONNX Runtime is not available. Please install it to use this framework.'
                )

    @with_exception_handling
    def _load_model(self, model_path: str) ->Any:
        """Load a model from the specified path."""
        logger.info(f'Loading {self.framework} model from {model_path}')
        try:
            if self.framework == 'tensorflow':
                return tf.saved_model.load(model_path)
            elif self.framework == 'pytorch':
                return torch.load(model_path)
            elif self.framework == 'onnx':
                return onnx.load(model_path)
            else:
                raise ValueError(f'Unsupported framework: {self.framework}')
        except Exception as e:
            logger.error(f'Error loading model: {str(e)}')
            raise

    def profile_model(self, input_data: Any, batch_sizes: List[int]=[1, 8, 
        32, 64], warmup_runs: int=10, profile_runs: int=100, profile_memory:
        bool=True, profile_cpu: bool=True, export_trace: bool=True) ->Dict[
        str, Any]:
        """
        Profile the model's performance characteristics.

        Args:
            input_data: Input data for profiling
            batch_sizes: List of batch sizes to profile
            warmup_runs: Number of warmup runs
            profile_runs: Number of profiling runs
            profile_memory: Whether to profile memory usage
            profile_cpu: Whether to profile CPU usage
            export_trace: Whether to export a trace file

        Returns:
            Dictionary with profiling results
        """
        logger.info(f'Profiling model with {len(batch_sizes)} batch sizes')
        if self.model_object is None:
            raise ValueError(
                'No model loaded. Please provide a model_path or model_object.'
                )
        results = {'model_name': self.model_name, 'framework': self.
            framework, 'timestamp': datetime.now().isoformat(),
            'batch_sizes': {}, 'summary': {}}
        for batch_size in batch_sizes:
            logger.info(f'Profiling batch size {batch_size}')
            batched_input = self._prepare_batch(input_data, batch_size)
            logger.info(f'Performing {warmup_runs} warmup runs')
            for _ in range(warmup_runs):
                _ = self._run_inference(batched_input)
            logger.info(f'Performing {profile_runs} profile runs')
            latencies = []
            memory_usage = []
            cpu_usage = []
            if profile_memory or profile_cpu:
                process = psutil.Process(os.getpid())
            for i in range(profile_runs):
                if profile_cpu:
                    cpu_start = process.cpu_percent()
                if profile_memory:
                    mem_start = process.memory_info().rss / (1024 * 1024)
                start_time = time.time()
                _ = self._run_inference(batched_input)
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                if profile_cpu:
                    cpu_end = process.cpu_percent()
                    cpu_usage.append(cpu_end - cpu_start if cpu_end >
                        cpu_start else cpu_end)
                if profile_memory:
                    mem_end = process.memory_info().rss / (1024 * 1024)
                    memory_usage.append(mem_end - mem_start if mem_end >
                        mem_start else 0)
                if (i + 1) % 10 == 0:
                    logger.info(
                        f'Completed {i + 1}/{profile_runs} profile runs')
            batch_results = {'latency_ms': {'mean': np.mean(latencies),
                'median': np.median(latencies), 'min': np.min(latencies),
                'max': np.max(latencies), 'p95': np.percentile(latencies, 
                95), 'p99': np.percentile(latencies, 99), 'std': np.std(
                latencies)}, 'throughput': {'samples_per_second': 
                batch_size * profile_runs / (sum(latencies) / 1000),
                'batches_per_second': profile_runs / (sum(latencies) / 1000)}}
            if profile_cpu:
                batch_results['cpu_usage_percent'] = {'mean': np.mean(
                    cpu_usage), 'max': np.max(cpu_usage)}
            if profile_memory:
                batch_results['memory_usage_mb'] = {'mean': np.mean(
                    memory_usage), 'max': np.max(memory_usage)}
            results['batch_sizes'][str(batch_size)] = batch_results
            if export_trace:
                trace_path = self._export_trace(batch_size)
                if trace_path:
                    batch_results['trace_path'] = str(trace_path)
        summary = {'optimal_batch_size': self._find_optimal_batch_size(
            results), 'max_throughput': max(results['batch_sizes'][str(bs)]
            ['throughput']['samples_per_second'] for bs in batch_sizes),
            'min_latency': min(results['batch_sizes'][str(bs)]['latency_ms'
            ]['mean'] for bs in batch_sizes)}
        results['summary'] = summary
        results_path = (self.output_dir /
            f'{self.model_name}_profiling_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.profiling_results = results
        logger.info(
            f"Profiling completed. Optimal batch size: {summary['optimal_batch_size']}"
            )
        return results

    @with_exception_handling
    def _prepare_batch(self, input_data: Any, batch_size: int) ->Any:
        """Prepare input data for the specified batch size."""
        try:
            if self.framework == 'tensorflow':
                if isinstance(input_data, np.ndarray):
                    if len(input_data.shape) == 1 or input_data.shape[0] == 1:
                        return np.repeat(input_data.reshape(1, -1),
                            batch_size, axis=0)
                    elif input_data.shape[0] >= batch_size:
                        return input_data[:batch_size]
                    else:
                        repeats = batch_size // input_data.shape[0] + 1
                        repeated = np.repeat(input_data, repeats, axis=0)
                        return repeated[:batch_size]
                elif isinstance(input_data, tf.Tensor):
                    if input_data.shape[0] == 1:
                        return tf.repeat(input_data, batch_size, axis=0)
                    elif input_data.shape[0] >= batch_size:
                        return input_data[:batch_size]
                    else:
                        repeats = batch_size // input_data.shape[0] + 1
                        repeated = tf.repeat(input_data, repeats, axis=0)
                        return repeated[:batch_size]
                else:
                    raise ValueError(
                        f'Unsupported input type for TensorFlow: {type(input_data)}'
                        )
            elif self.framework == 'pytorch':
                if isinstance(input_data, np.ndarray):
                    input_tensor = torch.from_numpy(input_data)
                    if input_tensor.shape[0] == 1:
                        return input_tensor.repeat(batch_size, *[(1) for _ in
                            range(len(input_tensor.shape) - 1)])
                    elif input_tensor.shape[0] >= batch_size:
                        return input_tensor[:batch_size]
                    else:
                        repeats = batch_size // input_tensor.shape[0] + 1
                        repeated = input_tensor.repeat(repeats, *[(1) for _ in
                            range(len(input_tensor.shape) - 1)])
                        return repeated[:batch_size]
                elif isinstance(input_data, torch.Tensor):
                    if input_data.shape[0] == 1:
                        return input_data.repeat(batch_size, *[(1) for _ in
                            range(len(input_data.shape) - 1)])
                    elif input_data.shape[0] >= batch_size:
                        return input_data[:batch_size]
                    else:
                        repeats = batch_size // input_data.shape[0] + 1
                        repeated = input_data.repeat(repeats, *[(1) for _ in
                            range(len(input_data.shape) - 1)])
                        return repeated[:batch_size]
                else:
                    raise ValueError(
                        f'Unsupported input type for PyTorch: {type(input_data)}'
                        )
            elif self.framework == 'onnx':
                if isinstance(input_data, np.ndarray):
                    if len(input_data.shape) == 1 or input_data.shape[0] == 1:
                        return np.repeat(input_data.reshape(1, -1),
                            batch_size, axis=0)
                    elif input_data.shape[0] >= batch_size:
                        return input_data[:batch_size]
                    else:
                        repeats = batch_size // input_data.shape[0] + 1
                        repeated = np.repeat(input_data, repeats, axis=0)
                        return repeated[:batch_size]
                else:
                    raise ValueError(
                        f'Unsupported input type for ONNX: {type(input_data)}')
            else:
                raise ValueError(f'Unsupported framework: {self.framework}')
        except Exception as e:
            logger.error(f'Error preparing batch: {str(e)}')
            raise

    @with_exception_handling
    def _run_inference(self, input_data: Any) ->Any:
        """Run inference with the model."""
        try:
            if self.framework == 'tensorflow':
                if hasattr(self.model_object, 'predict'):
                    return self.model_object.predict(input_data)
                else:
                    return self.model_object(input_data)
            elif self.framework == 'pytorch':
                if hasattr(self.model_object, 'eval'):
                    self.model_object.eval()
                with torch.no_grad():
                    return self.model_object(input_data)
            elif self.framework == 'onnx':
                session = ort.InferenceSession(self.model_path)
                input_name = session.get_inputs()[0].name
                return session.run(None, {input_name: input_data})
            else:
                raise ValueError(f'Unsupported framework: {self.framework}')
        except Exception as e:
            logger.error(f'Error running inference: {str(e)}')
            raise

    @with_exception_handling
    def _export_trace(self, batch_size: int) ->Optional[Path]:
        """Export a trace file for the model."""
        try:
            if self.framework == 'tensorflow':
                trace_path = (self.output_dir /
                    f'{self.model_name}_batch{batch_size}_trace')
                trace_path.mkdir(exist_ok=True, parents=True)
                input_shape = [batch_size, 10]
                input_data = np.random.randn(*input_shape).astype(np.float32)
                try:
                    tf.profiler.experimental.start(str(trace_path))
                    _ = self._run_inference(input_data)
                    tf.profiler.experimental.stop()
                except Exception as e:
                    logger.warning(f'Error using TensorFlow profiler: {str(e)}'
                        )
                return trace_path
            elif self.framework == 'pytorch':
                trace_path = (self.output_dir /
                    f'{self.model_name}_batch{batch_size}_trace.json')
                input_shape = [batch_size, 10]
                input_data = torch.randn(*input_shape)
                try:
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if torch.cuda.is_available():
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    with torch.profiler.profile(activities=activities,
                        record_shapes=True, profile_memory=True, with_stack
                        =True) as prof:
                        _ = self._run_inference(input_data)
                    prof.export_chrome_trace(str(trace_path))
                except Exception as e:
                    logger.warning(f'Error using PyTorch profiler: {str(e)}')
                    with open(trace_path, 'w') as f:
                        f.write('{}')
                return trace_path
            elif self.framework == 'onnx':
                return None
            else:
                raise ValueError(f'Unsupported framework: {self.framework}')
        except Exception as e:
            logger.error(f'Error exporting trace: {str(e)}')
            return None

    def _find_optimal_batch_size(self, results: Dict[str, Any]) ->int:
        """Find the optimal batch size based on throughput and latency."""
        batch_sizes = [int(bs) for bs in results['batch_sizes'].keys()]
        throughputs = [results['batch_sizes'][str(bs)]['throughput'][
            'samples_per_second'] for bs in batch_sizes]
        latencies = [results['batch_sizes'][str(bs)]['latency_ms']['mean'] for
            bs in batch_sizes]
        max_throughput_idx = np.argmax(throughputs)
        max_throughput_batch_size = batch_sizes[max_throughput_idx]
        latency_threshold = np.percentile(latencies, 95)
        acceptable_batch_sizes = [bs for bs, lat in zip(batch_sizes,
            latencies) if lat <= latency_threshold]
        if not acceptable_batch_sizes:
            min_latency_idx = np.argmin(latencies)
            return batch_sizes[min_latency_idx]
        acceptable_throughputs = [throughputs[batch_sizes.index(bs)] for bs in
            acceptable_batch_sizes]
        max_acceptable_throughput_idx = np.argmax(acceptable_throughputs)
        return acceptable_batch_sizes[max_acceptable_throughput_idx]

    @with_exception_handling
    def start_prometheus_exporter(self) ->None:
        """Start a Prometheus metrics exporter."""
        logger.info(
            f'Starting Prometheus exporter on port {self.prometheus_port}')
        try:
            import prometheus_client
        except ImportError:
            logger.error(
                'prometheus_client not available. Please install it to use this feature.'
                )
            return
        self.prom_inference_counter = prometheus_client.Counter(
            'ml_inference_count_total', 'Total number of inferences', [
            'model_name', 'framework'])
        self.prom_inference_latency = prometheus_client.Histogram(
            'ml_inference_latency_milliseconds',
            'Inference latency in milliseconds', ['model_name', 'framework',
            'batch_size'], buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 
            1000, 2000, 5000))
        self.prom_error_counter = prometheus_client.Counter(
            'ml_inference_errors_total', 'Total number of inference errors',
            ['model_name', 'framework'])
        self.prom_memory_gauge = prometheus_client.Gauge(
            'ml_memory_usage_megabytes', 'Memory usage in megabytes', [
            'model_name', 'framework'])
        self.prom_cpu_gauge = prometheus_client.Gauge('ml_cpu_usage_percent',
            'CPU usage in percent', ['model_name', 'framework'])
        prometheus_client.start_http_server(self.prometheus_port)
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info(
            f'Prometheus exporter started on http://localhost:{self.prometheus_port}{self.prometheus_endpoint}'
            )

    @with_exception_handling
    def _monitoring_loop(self) ->None:
        """Background thread for updating monitoring metrics."""
        process = psutil.Process(os.getpid())
        while self.is_monitoring:
            try:
                memory_mb = process.memory_info().rss / (1024 * 1024)
                self.prom_memory_gauge.labels(model_name=self.model_name,
                    framework=self.framework).set(memory_mb)
                cpu_percent = process.cpu_percent()
                self.prom_cpu_gauge.labels(model_name=self.model_name,
                    framework=self.framework).set(cpu_percent)
                time.sleep(1)
            except Exception as e:
                logger.error(f'Error in monitoring loop: {str(e)}')
                time.sleep(5)

    def stop_prometheus_exporter(self) ->None:
        """Stop the Prometheus metrics exporter."""
        logger.info('Stopping Prometheus exporter')
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info('Prometheus exporter stopped')

    @with_exception_handling
    def record_inference(self, input_data: Any, batch_size: Optional[int]=
        None, record_metrics: bool=True) ->Tuple[Any, float]:
        """
        Run inference and record metrics.

        Args:
            input_data: Input data for inference
            batch_size: Batch size (if None, inferred from input data)
            record_metrics: Whether to record metrics

        Returns:
            Tuple of (inference result, latency in ms)
        """
        if batch_size is None:
            if hasattr(input_data, 'shape') and len(input_data.shape) > 0:
                batch_size = input_data.shape[0]
            else:
                batch_size = 1
        start_time = time.time()
        try:
            result = self._run_inference(input_data)
            success = True
        except Exception as e:
            logger.error(f'Error during inference: {str(e)}')
            result = None
            success = False
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        if record_metrics:
            self.metrics['inference_count'] += 1
            self.metrics['inference_latency_ms'].append(latency_ms)
            self.metrics['batch_sizes'].append(batch_size)
            self.metrics['timestamp'].append(datetime.now().isoformat())
            if not success:
                self.metrics['error_count'] += 1
            try:
                import prometheus_client
                if hasattr(self, 'prom_inference_counter'):
                    self.prom_inference_counter.labels(model_name=self.
                        model_name, framework=self.framework).inc()
                    if success:
                        self.prom_inference_latency.labels(model_name=self.
                            model_name, framework=self.framework,
                            batch_size=str(batch_size)).observe(latency_ms)
                    else:
                        self.prom_error_counter.labels(model_name=self.
                            model_name, framework=self.framework).inc()
            except ImportError:
                pass
        return result, latency_ms

    def generate_grafana_dashboard(self, dashboard_title: str=
        'ML Model Performance', prometheus_datasource: str='Prometheus',
        dashboard_path: Optional[str]=None) ->Dict[str, Any]:
        """
        Generate a Grafana dashboard for monitoring the model.

        Args:
            dashboard_title: Title of the dashboard
            prometheus_datasource: Name of the Prometheus data source in Grafana
            dashboard_path: Path to save the dashboard JSON

        Returns:
            Dictionary with dashboard definition
        """
        logger.info(f'Generating Grafana dashboard: {dashboard_title}')
        dashboard = {'annotations': {'list': [{'builtIn': 1, 'datasource':
            '-- Grafana --', 'enable': True, 'hide': True, 'iconColor':
            'rgba(0, 211, 255, 1)', 'name': 'Annotations & Alerts', 'type':
            'dashboard'}]}, 'editable': True, 'gnetId': None,
            'graphTooltip': 0, 'id': None, 'links': [], 'panels': [],
            'refresh': '5s', 'schemaVersion': 22, 'style': 'dark', 'tags':
            ['ml', 'model', 'performance'], 'templating': {'list': []},
            'time': {'from': 'now-1h', 'to': 'now'}, 'timepicker': {
            'refresh_intervals': ['5s', '10s', '30s', '1m', '5m', '15m',
            '30m', '1h', '2h', '1d']}, 'timezone': '', 'title':
            dashboard_title, 'uid':
            f"ml-{self.model_name.lower().replace(' ', '-')}", 'version': 1}
        panels = []
        panels.append({'aliasColors': {}, 'bars': False, 'dashLength': 10,
            'dashes': False, 'datasource': prometheus_datasource, 'fill': 1,
            'fillGradient': 0, 'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 0},
            'hiddenSeries': False, 'id': 1, 'legend': {'avg': False,
            'current': False, 'max': False, 'min': False, 'show': True,
            'total': False, 'values': False}, 'lines': True, 'linewidth': 1,
            'nullPointMode': 'null', 'options': {'dataLinks': []},
            'percentage': False, 'pointradius': 2, 'points': False,
            'renderer': 'flot', 'seriesOverrides': [], 'spaceLength': 10,
            'stack': False, 'steppedLine': False, 'targets': [{'expr':
            f'rate(ml_inference_count_total{{model_name="{self.model_name}"}}[1m])'
            , 'interval': '', 'legendFormat': 'Inferences per second',
            'refId': 'A'}], 'thresholds': [], 'timeFrom': None,
            'timeRegions': [], 'timeShift': None, 'title': 'Inference Rate',
            'tooltip': {'shared': True, 'sort': 0, 'value_type':
            'individual'}, 'type': 'graph', 'xaxis': {'buckets': None,
            'mode': 'time', 'name': None, 'show': True, 'values': []},
            'yaxes': [{'format': 'short', 'label': 'Inferences/s',
            'logBase': 1, 'max': None, 'min': None, 'show': True}, {
            'format': 'short', 'label': '', 'logBase': 1, 'max': None,
            'min': None, 'show': True}], 'yaxis': {'align': False,
            'alignLevel': None}})
        panels.append({'aliasColors': {}, 'bars': False, 'dashLength': 10,
            'dashes': False, 'datasource': prometheus_datasource, 'fill': 1,
            'fillGradient': 0, 'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 0
            }, 'hiddenSeries': False, 'id': 2, 'legend': {'avg': False,
            'current': False, 'max': False, 'min': False, 'show': True,
            'total': False, 'values': False}, 'lines': True, 'linewidth': 1,
            'nullPointMode': 'null', 'options': {'dataLinks': []},
            'percentage': False, 'pointradius': 2, 'points': False,
            'renderer': 'flot', 'seriesOverrides': [], 'spaceLength': 10,
            'stack': False, 'steppedLine': False, 'targets': [{'expr':
            f'histogram_quantile(0.95, sum(rate(ml_inference_latency_milliseconds_bucket{{model_name="{self.model_name}"}}[1m])) by (le))'
            , 'interval': '', 'legendFormat': 'p95 Latency', 'refId': 'A'},
            {'expr':
            f'histogram_quantile(0.50, sum(rate(ml_inference_latency_milliseconds_bucket{{model_name="{self.model_name}"}}[1m])) by (le))'
            , 'interval': '', 'legendFormat': 'p50 Latency', 'refId': 'B'}],
            'thresholds': [], 'timeFrom': None, 'timeRegions': [],
            'timeShift': None, 'title': 'Inference Latency', 'tooltip': {
            'shared': True, 'sort': 0, 'value_type': 'individual'}, 'type':
            'graph', 'xaxis': {'buckets': None, 'mode': 'time', 'name':
            None, 'show': True, 'values': []}, 'yaxes': [{'format': 'ms',
            'label': 'Latency', 'logBase': 1, 'max': None, 'min': None,
            'show': True}, {'format': 'short', 'label': '', 'logBase': 1,
            'max': None, 'min': None, 'show': True}], 'yaxis': {'align': 
            False, 'alignLevel': None}})
        panels.append({'aliasColors': {}, 'bars': False, 'dashLength': 10,
            'dashes': False, 'datasource': prometheus_datasource, 'fill': 1,
            'fillGradient': 0, 'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 8},
            'hiddenSeries': False, 'id': 3, 'legend': {'avg': False,
            'current': False, 'max': False, 'min': False, 'show': True,
            'total': False, 'values': False}, 'lines': True, 'linewidth': 1,
            'nullPointMode': 'null', 'options': {'dataLinks': []},
            'percentage': False, 'pointradius': 2, 'points': False,
            'renderer': 'flot', 'seriesOverrides': [], 'spaceLength': 10,
            'stack': False, 'steppedLine': False, 'targets': [{'expr':
            f'rate(ml_inference_errors_total{{model_name="{self.model_name}"}}[1m])'
            , 'interval': '', 'legendFormat': 'Errors per second', 'refId':
            'A'}], 'thresholds': [], 'timeFrom': None, 'timeRegions': [],
            'timeShift': None, 'title': 'Error Rate', 'tooltip': {'shared':
            True, 'sort': 0, 'value_type': 'individual'}, 'type': 'graph',
            'xaxis': {'buckets': None, 'mode': 'time', 'name': None, 'show':
            True, 'values': []}, 'yaxes': [{'format': 'short', 'label':
            'Errors/s', 'logBase': 1, 'max': None, 'min': None, 'show': 
            True}, {'format': 'short', 'label': '', 'logBase': 1, 'max':
            None, 'min': None, 'show': True}], 'yaxis': {'align': False,
            'alignLevel': None}})
        panels.append({'aliasColors': {}, 'bars': False, 'dashLength': 10,
            'dashes': False, 'datasource': prometheus_datasource, 'fill': 1,
            'fillGradient': 0, 'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 8
            }, 'hiddenSeries': False, 'id': 4, 'legend': {'avg': False,
            'current': False, 'max': False, 'min': False, 'show': True,
            'total': False, 'values': False}, 'lines': True, 'linewidth': 1,
            'nullPointMode': 'null', 'options': {'dataLinks': []},
            'percentage': False, 'pointradius': 2, 'points': False,
            'renderer': 'flot', 'seriesOverrides': [], 'spaceLength': 10,
            'stack': False, 'steppedLine': False, 'targets': [{'expr':
            f'ml_memory_usage_megabytes{{model_name="{self.model_name}"}}',
            'interval': '', 'legendFormat': 'Memory (MB)', 'refId': 'A'}, {
            'expr':
            f'ml_cpu_usage_percent{{model_name="{self.model_name}"}}',
            'interval': '', 'legendFormat': 'CPU (%)', 'refId': 'B'}],
            'thresholds': [], 'timeFrom': None, 'timeRegions': [],
            'timeShift': None, 'title': 'Resource Usage', 'tooltip': {
            'shared': True, 'sort': 0, 'value_type': 'individual'}, 'type':
            'graph', 'xaxis': {'buckets': None, 'mode': 'time', 'name':
            None, 'show': True, 'values': []}, 'yaxes': [{'format': 'short',
            'label': '', 'logBase': 1, 'max': None, 'min': None, 'show': 
            True}, {'format': 'short', 'label': '', 'logBase': 1, 'max':
            None, 'min': None, 'show': True}], 'yaxis': {'align': False,
            'alignLevel': None}})
        dashboard['panels'] = panels
        if dashboard_path:
            dashboard_file = Path(dashboard_path)
            with open(dashboard_file, 'w') as f:
                json.dump(dashboard, f, indent=2)
            logger.info(f'Dashboard saved to {dashboard_file}')
        return dashboard

    def configure_alerts(self, latency_threshold_ms: float=100.0,
        error_rate_threshold: float=0.01, memory_threshold_mb: float=1000.0,
        cpu_threshold_percent: float=80.0, alert_config_path: Optional[str]
        =None) ->Dict[str, Any]:
        """
        Configure alerts for the model.

        Args:
            latency_threshold_ms: Threshold for latency alerts
            error_rate_threshold: Threshold for error rate alerts
            memory_threshold_mb: Threshold for memory usage alerts
            cpu_threshold_percent: Threshold for CPU usage alerts
            alert_config_path: Path to save the alert configuration

        Returns:
            Dictionary with alert configuration
        """
        logger.info('Configuring alerts')
        alerts = {'groups': [{'name': f'ml_model_{self.model_name}_alerts',
            'rules': [{'alert': 'HighLatency', 'expr':
            f'histogram_quantile(0.95, sum(rate(ml_inference_latency_milliseconds_bucket{{model_name="{self.model_name}"}}[5m])) by (le)) > {latency_threshold_ms}'
            , 'for': '1m', 'labels': {'severity': 'warning'}, 'annotations':
            {'summary': f'High latency for model {self.model_name}',
            'description':
            f'95th percentile latency is above {latency_threshold_ms}ms for model {self.model_name}'
            }}, {'alert': 'HighErrorRate', 'expr':
            f'rate(ml_inference_errors_total{{model_name="{self.model_name}"}}[5m]) / rate(ml_inference_count_total{{model_name="{self.model_name}"}}[5m]) > {error_rate_threshold}'
            , 'for': '1m', 'labels': {'severity': 'warning'}, 'annotations':
            {'summary': f'High error rate for model {self.model_name}',
            'description':
            f'Error rate is above {error_rate_threshold * 100}% for model {self.model_name}'
            }}, {'alert': 'HighMemoryUsage', 'expr':
            f'ml_memory_usage_megabytes{{model_name="{self.model_name}"}} > {memory_threshold_mb}'
            , 'for': '5m', 'labels': {'severity': 'warning'}, 'annotations':
            {'summary': f'High memory usage for model {self.model_name}',
            'description':
            f'Memory usage is above {memory_threshold_mb}MB for model {self.model_name}'
            }}, {'alert': 'HighCPUUsage', 'expr':
            f'ml_cpu_usage_percent{{model_name="{self.model_name}"}} > {cpu_threshold_percent}'
            , 'for': '5m', 'labels': {'severity': 'warning'}, 'annotations':
            {'summary': f'High CPU usage for model {self.model_name}',
            'description':
            f'CPU usage is above {cpu_threshold_percent}% for model {self.model_name}'
            }}]}]}
        if alert_config_path:
            alert_file = Path(alert_config_path)
            with open(alert_file, 'w') as f:
                json.dump(alerts, f, indent=2)
            logger.info(f'Alert configuration saved to {alert_file}')
        return alerts
