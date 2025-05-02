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
from ml_integration_service.services.job_tracking import JobTrackingService
from ml_integration_service.services.model_registry import ModelRegistryService

logger = get_logger(__name__)

# Define Prometheus metrics
ML_JOBS_ACTIVE = Gauge(
    'forex_ml_jobs_active_count', 
    'Number of currently active ML retraining jobs',
    ['job_type']
)

ML_JOBS_COMPLETED = Counter(
    'forex_ml_jobs_completed_count',
    'Total number of completed ML retraining jobs',
    ['model_id', 'job_type', 'result']
)

ML_JOBS_FAILED = Counter(
    'forex_ml_jobs_failed_count',
    'Total number of failed ML retraining jobs',
    ['model_id', 'job_type', 'failure_reason']
)

ML_JOB_EXECUTION_TIME = Histogram(
    'forex_ml_job_execution_time_seconds',
    'ML job execution time in seconds',
    ['model_id', 'job_type'],
    buckets=(60, 300, 600, 1800, 3600, 7200, 10800, 14400)  # 1m, 5m, 10m, 30m, 1h, 2h, 3h, 4h
)

ML_JOB_SUCCESS_RATE = Gauge(
    'forex_ml_job_success_rate',
    'Success rate of ML retraining jobs',
    ['model_id']
)

MODEL_PERFORMANCE = Gauge(
    'forex_model_performance',
    'Performance metrics for ML models',
    ['model_id', 'metric_name']
)


class MLMetricsExporter:
    """
    Exports metrics from the ML Integration Service to the monitoring system.
    """
    
    def __init__(
        self, 
        job_tracking_service: JobTrackingService,
        model_registry_service: ModelRegistryService,
        config_manager: Optional[ConfigurationManager] = None
    ):
        """
        Initialize the ML metrics exporter.
        
        Args:
            job_tracking_service: Service for tracking ML jobs
            model_registry_service: Service for accessing model registry
            config_manager: Configuration manager
        """
        self.job_tracking_service = job_tracking_service
        self.model_registry_service = model_registry_service
        self.config_manager = config_manager
        
        # Load configuration
        self.config = self._load_config()
        
        # Metrics export settings
        self.export_interval = self.config.get("export_interval_seconds", 60)
        self.exporter_port = self.config.get("exporter_port", 9101)
        
        # Track whether the exporter is running
        self.is_running = False
        self.exporter_task = None
        
        logger.info(f"ML Metrics Exporter initialized with port {self.exporter_port} "
                   f"and interval {self.export_interval}s")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from the configuration manager or use defaults.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        default_config = {
            "export_interval_seconds": 60,
            "exporter_port": 9101,
            "metrics_enabled": True
        }
        
        if self.config_manager:
            try:
                metrics_config = self.config_manager.get_config("metrics_exporter")
                if metrics_config:
                    return {**default_config, **metrics_config}
            except Exception as e:
                logger.warning(f"Failed to load metrics exporter config: {e}")
                
        return default_config
    
    def start(self, port: Optional[int] = None) -> None:
        """
        Start the metrics exporter HTTP server and metrics collection task.
        
        Args:
            port: Optional override for the exporter port
        """
        if self.is_running:
            logger.warning("Metrics exporter is already running")
            return
            
        # Use provided port or default from config
        exporter_port = port or self.exporter_port
        
        try:
            # Start Prometheus HTTP server
            start_http_server(exporter_port)
            logger.info(f"Started metrics HTTP server on port {exporter_port}")
            
            # Start metrics collection task
            self.exporter_task = asyncio.create_task(self._collect_metrics_periodically())
            self.is_running = True
            
            logger.info("ML metrics exporter started successfully")
        except Exception as e:
            logger.error(f"Failed to start metrics exporter: {e}")
            raise
    
    async def stop(self) -> None:
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
        logger.info("ML metrics exporter stopped")
    
    async def _collect_metrics_periodically(self) -> None:
        """Collect metrics at regular intervals."""
        try:
            while True:
                await self._collect_and_export_metrics()
                await asyncio.sleep(self.export_interval)
        except asyncio.CancelledError:
            logger.info("Metrics collection task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in metrics collection: {e}")
            raise
    
    async def _collect_and_export_metrics(self) -> None:
        """Collect current metrics and update exporters."""
        try:
            # Get active jobs
            active_jobs = await self.job_tracking_service.get_active_jobs()
            
            # Reset active job gauges
            ML_JOBS_ACTIVE.labels(job_type="retraining").set(0)
            ML_JOBS_ACTIVE.labels(job_type="evaluation").set(0)
            ML_JOBS_ACTIVE.labels(job_type="deployment").set(0)
            
            # Count active jobs by type
            job_counts = {"retraining": 0, "evaluation": 0, "deployment": 0}
            for job in active_jobs:
                job_type = job.get("job_type", "retraining")
                if job_type in job_counts:
                    job_counts[job_type] += 1
            
            # Update active job metrics
            for job_type, count in job_counts.items():
                ML_JOBS_ACTIVE.labels(job_type=job_type).set(count)
            
            # Get recently completed jobs (in the last reporting period)
            completed_jobs = await self.job_tracking_service.get_recently_completed_jobs(
                seconds=self.export_interval * 2  # Look back a bit further than the interval
            )
            
            # Update completed and failed job metrics
            for job in completed_jobs:
                job_type = job.get("job_type", "retraining")
                model_id = job.get("model_id", "unknown")
                
                # Track execution time
                start_time = job.get("start_time")
                end_time = job.get("end_time")
                if start_time and end_time:
                    try:
                        # Convert to datetime objects if they're strings
                        if isinstance(start_time, str):
                            start_time = datetime.fromisoformat(start_time)
                        if isinstance(end_time, str):
                            end_time = datetime.fromisoformat(end_time)
                        
                        execution_time = (end_time - start_time).total_seconds()
                        ML_JOB_EXECUTION_TIME.labels(
                            model_id=model_id, 
                            job_type=job_type
                        ).observe(execution_time)
                    except Exception as e:
                        logger.warning(f"Could not calculate execution time for job {job.get('job_id')}: {e}")
                
                # Track success/failure
                status = job.get("status", "").lower()
                if status in ["completed", "succeeded", "success"]:
                    ML_JOBS_COMPLETED.labels(
                        model_id=model_id,
                        job_type=job_type,
                        result="success"
                    ).inc()
                else:
                    failure_reason = job.get("failure_reason", "unknown")
                    ML_JOBS_FAILED.labels(
                        model_id=model_id,
                        job_type=job_type,
                        failure_reason=failure_reason
                    ).inc()
                    ML_JOBS_COMPLETED.labels(
                        model_id=model_id,
                        job_type=job_type,
                        result="failure"
                    ).inc()
            
            # Update model success rates and performance metrics
            models = await self.model_registry_service.list_models()
            for model in models:
                model_id = model.get("model_id")
                if not model_id:
                    continue
                
                # Calculate success rate
                model_jobs = await self.job_tracking_service.get_jobs_for_model(model_id, limit=20)
                if model_jobs:
                    success_count = sum(1 for j in model_jobs 
                                       if j.get("status", "").lower() in ["completed", "succeeded", "success"])
                    success_rate = success_count / len(model_jobs)
                    ML_JOB_SUCCESS_RATE.labels(model_id=model_id).set(success_rate)
                
                # Export model performance metrics
                metrics = model.get("performance_metrics", {})
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        MODEL_PERFORMANCE.labels(
                            model_id=model_id,
                            metric_name=metric_name
                        ).set(value)
        
        except Exception as e:
            logger.error(f"Error collecting ML metrics: {e}", exc_info=True)


def create_metrics_exporter(
    job_tracking_service: JobTrackingService,
    model_registry_service: ModelRegistryService,
    config_manager: Optional[ConfigurationManager] = None
) -> MLMetricsExporter:
    """
    Create and initialize the ML metrics exporter.
    
    Args:
        job_tracking_service: Service for tracking ML jobs
        model_registry_service: Service for accessing model registry
        config_manager: Configuration manager
        
    Returns:
        MLMetricsExporter: Initialized metrics exporter
    """
    exporter = MLMetricsExporter(
        job_tracking_service=job_tracking_service,
        model_registry_service=model_registry_service,
        config_manager=config_manager
    )
    return exporter
