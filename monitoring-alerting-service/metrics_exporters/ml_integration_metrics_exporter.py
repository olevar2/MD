"""
ML Integration Service Metrics Exporter

This module exports metrics from the ML Integration Service to the monitoring system,
enabling dashboards and alerts to track ML retraining jobs, performance, and latency.
"""
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from prometheus_client import Counter, Gauge, Histogram, Summary
from prometheus_client.exposition import start_http_server
from core_foundations.utils.logger import get_logger
from core_foundations.config.configuration import ConfigurationManager
from monitoring_alerting_service.adapters.ml_integration_adapter import MLIntegrationMonitoringAdapter
logger = get_logger(__name__)
ML_JOBS_ACTIVE = Gauge('forex_ml_jobs_active_count',
    'Number of currently active ML retraining jobs', ['job_type'])
ML_JOBS_COMPLETED = Counter('forex_ml_jobs_completed_count',
    'Total number of completed ML retraining jobs', ['model_id', 'job_type',
    'result'])
ML_JOBS_FAILED = Counter('forex_ml_jobs_failed_count',
    'Total number of failed ML retraining jobs', ['model_id', 'job_type',
    'failure_reason'])
ML_JOB_EXECUTION_TIME = Histogram('forex_ml_job_execution_time_seconds',
    'ML job execution time in seconds', ['model_id', 'job_type'], buckets=(
    60, 300, 600, 1800, 3600, 7200, 10800, 14400))
ML_JOB_SUCCESS_RATE = Gauge('forex_ml_job_success_rate',
    'Success rate of ML retraining jobs', ['model_id'])
MODEL_PERFORMANCE = Gauge('forex_model_performance',
    'Performance metrics for ML models', ['model_id', 'metric_name'])


from monitoring_alerting_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class MLMetricsExporter:
    """
    Exports metrics from the ML Integration Service to the monitoring system.
    """

    def __init__(self, ml_integration_adapter:
        MLIntegrationMonitoringAdapter, config_manager: Optional[
        ConfigurationManager]=None):
        """
        Initialize the ML metrics exporter.

        Args:
            ml_integration_adapter: Adapter for ML Integration Service
            config_manager: Configuration manager
        """
        self.ml_integration_adapter = ml_integration_adapter
        self.config_manager = config_manager
        self.config = self._load_config()
        self.export_interval = self.config_manager.get('export_interval_seconds', 60)
        self.exporter_port = self.config_manager.get('exporter_port', 9101)
        self.is_running = False
        self.exporter_task = None
        logger.info(
            f'ML Metrics Exporter initialized with port {self.exporter_port} and interval {self.export_interval}s'
            )

    @with_exception_handling
    def _load_config(self) ->Dict[str, Any]:
        """
        Load configuration from the configuration manager or use defaults.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        default_config = {'export_interval_seconds': 60, 'exporter_port': 
            9101, 'metrics_enabled': True}
        if self.config_manager:
            try:
                metrics_config = self.config_manager.get_config(
                    'metrics_exporter')
                if metrics_config:
                    return {**default_config, **metrics_config}
            except Exception as e:
                logger.warning(f'Failed to load metrics exporter config: {e}')
        return default_config

    @with_exception_handling
    def start(self, port: Optional[int]=None) ->None:
        """
        Start the metrics exporter HTTP server and metrics collection task.

        Args:
            port: Optional override for the exporter port
        """
        if self.is_running:
            logger.warning('Metrics exporter is already running')
            return
        exporter_port = port or self.exporter_port
        try:
            start_http_server(exporter_port)
            logger.info(f'Started metrics HTTP server on port {exporter_port}')
            self.exporter_task = asyncio.create_task(self.
                _collect_metrics_periodically())
            self.is_running = True
            logger.info('ML metrics exporter started successfully')
        except Exception as e:
            logger.error(f'Failed to start metrics exporter: {e}')
            raise

    @async_with_exception_handling
    async def stop(self) ->None:
        """Stop the metrics collection task."""
        if not self.is_running:
            return
        if self.exporter_task and not self.exporter_task.done():
            self.exporter_task.cancel()
            try:
                await self.exporter_task
            except asyncio.CancelledError:
                pass
        self.is_running = False
        logger.info('ML metrics exporter stopped')

    @async_with_exception_handling
    async def _collect_metrics_periodically(self) ->None:
        """Collect metrics at regular intervals."""
        try:
            while True:
                await self._collect_and_export_metrics()
                await asyncio.sleep(self.export_interval)
        except asyncio.CancelledError:
            logger.info('Metrics collection task cancelled')
            raise
        except Exception as e:
            logger.error(f'Error in metrics collection: {e}')
            raise

    @async_with_exception_handling
    async def _collect_and_export_metrics(self) ->None:
        """Collect current metrics and update exporters."""
        try:
            job_metrics = (await self.ml_integration_adapter.
                get_job_status_metrics())
            ML_JOBS_ACTIVE.labels(job_type='retraining').set(0)
            ML_JOBS_ACTIVE.labels(job_type='evaluation').set(0)
            ML_JOBS_ACTIVE.labels(job_type='deployment').set(0)
            status_counts = job_metrics.get('status_counts', {})
            for job_type, count in job_metrics.get('type_counts', {}).items():
                if job_type in ['retraining', 'evaluation', 'deployment']:
                    ML_JOBS_ACTIVE.labels(job_type=job_type).set(count)
            model_metrics = (await self.ml_integration_adapter.
                get_model_status_metrics())
            for model in model_metrics.get('models', []):
                model_id = model.get('model_id', 'unknown')
                try:
                    model_performance = (await self.ml_integration_adapter.
                        get_model_metrics(model_id=model_id))
                    metrics = model_performance.get('metrics', {})
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            MODEL_PERFORMANCE.labels(model_id=model_id,
                                metric_name=metric_name).set(value)
                    if 'success_rate' in metrics:
                        ML_JOB_SUCCESS_RATE.labels(model_id=model_id).set(
                            metrics['success_rate'])
                except Exception as e:
                    logger.warning(
                        f'Error getting metrics for model {model_id}: {e}')
            service_metrics = (await self.ml_integration_adapter.
                get_model_metrics())
            job_execution_metrics = service_metrics.get('job_execution', {})
            for job_id, job_data in job_execution_metrics.items():
                model_id = job_data.get('model_id', 'unknown')
                job_type = job_data.get('job_type', 'retraining')
                execution_time = job_data.get('execution_time')
                status = job_data.get('status', '').lower()
                if execution_time and isinstance(execution_time, (int, float)):
                    ML_JOB_EXECUTION_TIME.labels(model_id=model_id,
                        job_type=job_type).observe(execution_time)
                if status in ['completed', 'succeeded', 'success']:
                    ML_JOBS_COMPLETED.labels(model_id=model_id, job_type=
                        job_type, result='success').inc()
                elif status in ['failed', 'failure', 'error']:
                    failure_reason = job_data.get('failure_reason', 'unknown')
                    ML_JOBS_FAILED.labels(model_id=model_id, job_type=
                        job_type, failure_reason=failure_reason).inc()
                    ML_JOBS_COMPLETED.labels(model_id=model_id, job_type=
                        job_type, result='failure').inc()
        except Exception as e:
            logger.error(f'Error collecting ML metrics: {e}', exc_info=True)


def create_metrics_exporter(config: Optional[Dict[str, Any]]=None,
    config_manager: Optional[ConfigurationManager]=None) ->MLMetricsExporter:
    """
    Create and initialize the ML metrics exporter.

    Args:
        config: Optional configuration dictionary
        config_manager: Configuration manager

    Returns:
        MLMetricsExporter: Initialized metrics exporter
    """
    ml_integration_adapter = MLIntegrationMonitoringAdapter(config=config)
    exporter = MLMetricsExporter(ml_integration_adapter=
        ml_integration_adapter, config_manager=config_manager)
    return exporter
