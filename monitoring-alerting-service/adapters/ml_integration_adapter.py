"""
ML Integration Service Adapter for Monitoring and Alerting Service.

This module provides adapter implementations for the ML Integration Service interfaces,
allowing the Monitoring and Alerting Service to interact with the ML Integration Service
without direct dependencies.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from common_lib.interfaces.ml_integration import (
    IMLModelRegistry, IMLJobTracker, IMLMetricsProvider,
    ModelStatus, ModelType, ModelFramework
)
from common_lib.adapters.ml_integration_adapter import (
    MLModelRegistryAdapter, MLJobTrackerAdapter, MLMetricsProviderAdapter
)
from common_lib.service_client.base_client import ServiceClientConfig
from common_lib.adapters.factory import AdapterFactory

logger = logging.getLogger(__name__)


class MLIntegrationMonitoringAdapter:
    """
    Adapter for ML Integration Service monitoring operations.
    
    This adapter provides methods for monitoring ML models, jobs, and metrics
    from the ML Integration Service.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adapter.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Create service client configuration
        service_config = ServiceClientConfig(
            base_url=self.config_manager.get('ml_integration_api_url', "http://ml-integration-service:8000/api/v1"),
            timeout=self.config_manager.get('timeout', 30),
            retry_attempts=self.config_manager.get('retry_attempts', 3),
            retry_backoff=self.config_manager.get('retry_backoff', 1.5)
        )
        
        # Create adapter factory
        adapter_factory = AdapterFactory(
            config_provider={"ml-integration-service": service_config}
        )
        
        # Create adapters
        self.model_registry = adapter_factory.create_ml_model_registry()
        self.job_tracker = adapter_factory.create_ml_job_tracker()
        self.metrics_provider = adapter_factory.create_ml_metrics_provider()
    
    async def get_model_metrics(
        self,
        model_id: Optional[str] = None,
        metric_types: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get metrics for ML models.
        
        Args:
            model_id: Optional model ID to filter by
            metric_types: Optional list of metric types to retrieve
            start_time: Optional start time for metrics
            end_time: Optional end time for metrics
            
        Returns:
            Dictionary containing model metrics
        """
        if model_id:
            # Get metrics for a specific model
            return await self.metrics_provider.get_model_metrics(
                model_id=model_id,
                metric_types=metric_types,
                start_time=start_time,
                end_time=end_time
            )
        else:
            # Get service-level metrics
            return await self.metrics_provider.get_service_metrics(
                metric_types=metric_types,
                start_time=start_time,
                end_time=end_time
            )
    
    async def get_job_status_metrics(
        self,
        job_type: Optional[str] = None,
        status: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get metrics for ML jobs.
        
        Args:
            job_type: Optional job type to filter by
            status: Optional job status to filter by
            start_date: Optional start date for jobs
            end_date: Optional end date for jobs
            
        Returns:
            Dictionary containing job metrics
        """
        # Get jobs
        jobs = await self.job_tracker.list_jobs(
            job_type=job_type,
            status=status,
            start_date=start_date,
            end_date=end_date
        )
        
        # Calculate metrics
        total_jobs = len(jobs)
        status_counts = {}
        type_counts = {}
        
        for job in jobs:
            # Count by status
            job_status = job.get("status", "unknown")
            if job_status not in status_counts:
                status_counts[job_status] = 0
            status_counts[job_status] += 1
            
            # Count by type
            job_type = job.get("job_type", "unknown")
            if job_type not in type_counts:
                type_counts[job_type] = 0
            type_counts[job_type] += 1
        
        # Calculate average duration
        durations = []
        for job in jobs:
            if job.get("start_time") and job.get("end_time"):
                start = datetime.fromisoformat(job["start_time"])
                end = datetime.fromisoformat(job["end_time"])
                duration = (end - start).total_seconds()
                durations.append(duration)
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_jobs": total_jobs,
            "status_counts": status_counts,
            "type_counts": type_counts,
            "average_duration_seconds": avg_duration
        }
    
    async def get_model_status_metrics(
        self,
        model_type: Optional[Union[str, ModelType]] = None,
        status: Optional[Union[str, ModelStatus]] = None,
        framework: Optional[Union[str, ModelFramework]] = None
    ) -> Dict[str, Any]:
        """
        Get metrics for ML models.
        
        Args:
            model_type: Optional model type to filter by
            status: Optional model status to filter by
            framework: Optional model framework to filter by
            
        Returns:
            Dictionary containing model metrics
        """
        # Get models
        models = await self.model_registry.list_models(
            model_type=model_type,
            status=status,
            framework=framework
        )
        
        # Calculate metrics
        total_models = len(models)
        status_counts = {}
        type_counts = {}
        framework_counts = {}
        
        for model in models:
            # Count by status
            model_status = model.get("status", "unknown")
            if model_status not in status_counts:
                status_counts[model_status] = 0
            status_counts[model_status] += 1
            
            # Count by type
            model_type = model.get("model_type", "unknown")
            if model_type not in type_counts:
                type_counts[model_type] = 0
            type_counts[model_type] += 1
            
            # Count by framework
            model_framework = model.get("framework", "unknown")
            if model_framework not in framework_counts:
                framework_counts[model_framework] = 0
            framework_counts[model_framework] += 1
        
        return {
            "total_models": total_models,
            "status_counts": status_counts,
            "type_counts": type_counts,
            "framework_counts": framework_counts
        }
