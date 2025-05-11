"""
ML Integration Service Adapter.

This module provides adapter implementations for the ML Integration Service interfaces.
These adapters allow other services to interact with the ML Integration Service
without direct dependencies, breaking circular dependencies.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, cast

import aiohttp
import json

from common_lib.interfaces.ml_integration import (
    IMLModelRegistry, IMLJobTracker, IMLModelDeployment, IMLMetricsProvider,
    ModelStatus, ModelType, ModelFramework
)
from common_lib.service_client.base_client import ServiceClientConfig
from common_lib.service_client.http_client import AsyncHTTPServiceClient
from common_lib.errors.base_exceptions import (
    ServiceUnavailableError, ResourceNotFoundError, ValidationError
)

logger = logging.getLogger(__name__)


class MLModelRegistryAdapter(IMLModelRegistry):
    """Adapter for ML Model Registry operations."""
    
    def __init__(self, client: Optional[AsyncHTTPServiceClient] = None, config: Optional[ServiceClientConfig] = None):
        """
        Initialize the adapter.
        
        Args:
            client: Optional pre-configured HTTP client
            config: Optional client configuration
        """
        self.client = client or AsyncHTTPServiceClient(
            config or ServiceClientConfig(
                base_url="http://ml-integration-service:8000/api/v1",
                timeout=30
            )
        )
    
    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_id: The ID of the model
            
        Returns:
            Dictionary containing model information
        """
        try:
            response = await self.client.get(f"/models/{model_id}")
            return response
        except ResourceNotFoundError:
            logger.error(f"Model with ID {model_id} not found")
            raise
        except Exception as e:
            logger.error(f"Error getting model info for {model_id}: {str(e)}")
            raise
    
    async def list_models(
        self, 
        model_type: Optional[Union[str, ModelType]] = None,
        status: Optional[Union[str, ModelStatus]] = None,
        framework: Optional[Union[str, ModelFramework]] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List models with optional filtering.
        
        Args:
            model_type: Filter by model type
            status: Filter by model status
            framework: Filter by model framework
            tags: Filter by model tags
            
        Returns:
            List of dictionaries containing model information
        """
        params = {}
        if model_type:
            params["model_type"] = model_type.value if isinstance(model_type, ModelType) else model_type
        if status:
            params["status"] = status.value if isinstance(status, ModelStatus) else status
        if framework:
            params["framework"] = framework.value if isinstance(framework, ModelFramework) else framework
        if tags:
            params["tags"] = ",".join(tags)
        
        try:
            response = await self.client.get("/models", params=params)
            return response.get("models", [])
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            raise
    
    async def register_model(
        self,
        name: str,
        version: str,
        model_type: Union[str, ModelType],
        framework: Union[str, ModelFramework],
        metadata: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Register a new model.
        
        Args:
            name: Model name
            version: Model version
            model_type: Type of model
            framework: Model framework
            metadata: Model metadata
            tags: Optional model tags
            
        Returns:
            The ID of the registered model
        """
        payload = {
            "name": name,
            "version": version,
            "model_type": model_type.value if isinstance(model_type, ModelType) else model_type,
            "framework": framework.value if isinstance(framework, ModelFramework) else framework,
            "metadata": metadata,
            "tags": tags or []
        }
        
        try:
            response = await self.client.post("/models", json=payload)
            return response.get("model_id", "")
        except ValidationError:
            logger.error(f"Validation error when registering model {name} {version}")
            raise
        except Exception as e:
            logger.error(f"Error registering model {name} {version}: {str(e)}")
            raise
    
    async def update_model_status(
        self,
        model_id: str,
        status: Union[str, ModelStatus],
        status_message: Optional[str] = None
    ) -> bool:
        """
        Update the status of a model.
        
        Args:
            model_id: The ID of the model
            status: New model status
            status_message: Optional status message
            
        Returns:
            True if the update was successful, False otherwise
        """
        payload = {
            "status": status.value if isinstance(status, ModelStatus) else status,
            "status_message": status_message
        }
        
        try:
            response = await self.client.patch(f"/models/{model_id}/status", json=payload)
            return response.get("success", False)
        except ResourceNotFoundError:
            logger.error(f"Model with ID {model_id} not found")
            raise
        except Exception as e:
            logger.error(f"Error updating model status for {model_id}: {str(e)}")
            raise


class MLJobTrackerAdapter(IMLJobTracker):
    """Adapter for ML Job Tracker operations."""
    
    def __init__(self, client: Optional[AsyncHTTPServiceClient] = None, config: Optional[ServiceClientConfig] = None):
        """
        Initialize the adapter.
        
        Args:
            client: Optional pre-configured HTTP client
            config: Optional client configuration
        """
        self.client = client or AsyncHTTPServiceClient(
            config or ServiceClientConfig(
                base_url="http://ml-integration-service:8000/api/v1",
                timeout=30
            )
        )
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a specific job.
        
        Args:
            job_id: The ID of the job
            
        Returns:
            Dictionary containing job status information
        """
        try:
            response = await self.client.get(f"/jobs/{job_id}")
            return response
        except ResourceNotFoundError:
            logger.error(f"Job with ID {job_id} not found")
            raise
        except Exception as e:
            logger.error(f"Error getting job status for {job_id}: {str(e)}")
            raise
    
    async def list_jobs(
        self,
        job_type: Optional[str] = None,
        status: Optional[str] = None,
        model_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        List jobs with optional filtering.
        
        Args:
            job_type: Filter by job type
            status: Filter by job status
            model_id: Filter by model ID
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            List of dictionaries containing job information
        """
        params = {}
        if job_type:
            params["job_type"] = job_type
        if status:
            params["status"] = status
        if model_id:
            params["model_id"] = model_id
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        
        try:
            response = await self.client.get("/jobs", params=params)
            return response.get("jobs", [])
        except Exception as e:
            logger.error(f"Error listing jobs: {str(e)}")
            raise
    
    async def create_job(
        self,
        job_type: str,
        model_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        priority: Optional[int] = None
    ) -> str:
        """
        Create a new job.
        
        Args:
            job_type: Type of job
            model_id: Optional model ID
            parameters: Optional job parameters
            priority: Optional job priority
            
        Returns:
            The ID of the created job
        """
        payload = {
            "job_type": job_type,
            "parameters": parameters or {}
        }
        
        if model_id:
            payload["model_id"] = model_id
        if priority is not None:
            payload["priority"] = priority
        
        try:
            response = await self.client.post("/jobs", json=payload)
            return response.get("job_id", "")
        except ValidationError:
            logger.error(f"Validation error when creating job of type {job_type}")
            raise
        except Exception as e:
            logger.error(f"Error creating job of type {job_type}: {str(e)}")
            raise
    
    async def update_job_status(
        self,
        job_id: str,
        status: str,
        status_message: Optional[str] = None,
        progress: Optional[float] = None
    ) -> bool:
        """
        Update the status of a job.
        
        Args:
            job_id: The ID of the job
            status: New job status
            status_message: Optional status message
            progress: Optional progress percentage (0.0 to 1.0)
            
        Returns:
            True if the update was successful, False otherwise
        """
        payload = {
            "status": status
        }
        
        if status_message:
            payload["status_message"] = status_message
        if progress is not None:
            payload["progress"] = progress
        
        try:
            response = await self.client.patch(f"/jobs/{job_id}/status", json=payload)
            return response.get("success", False)
        except ResourceNotFoundError:
            logger.error(f"Job with ID {job_id} not found")
            raise
        except Exception as e:
            logger.error(f"Error updating job status for {job_id}: {str(e)}")
            raise


class MLModelDeploymentAdapter(IMLModelDeployment):
    """Adapter for ML Model Deployment operations."""
    
    def __init__(self, client: Optional[AsyncHTTPServiceClient] = None, config: Optional[ServiceClientConfig] = None):
        """
        Initialize the adapter.
        
        Args:
            client: Optional pre-configured HTTP client
            config: Optional client configuration
        """
        self.client = client or AsyncHTTPServiceClient(
            config or ServiceClientConfig(
                base_url="http://ml-integration-service:8000/api/v1",
                timeout=30
            )
        )
    
    async def deploy_model(
        self,
        model_id: str,
        deployment_target: str,
        deployment_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Deploy a model to a specific target.
        
        Args:
            model_id: The ID of the model to deploy
            deployment_target: Target environment for deployment
            deployment_config: Optional deployment configuration
            
        Returns:
            Dictionary containing deployment information
        """
        payload = {
            "model_id": model_id,
            "deployment_target": deployment_target,
            "deployment_config": deployment_config or {}
        }
        
        try:
            response = await self.client.post("/deployments", json=payload)
            return response
        except ResourceNotFoundError:
            logger.error(f"Model with ID {model_id} not found")
            raise
        except ValidationError:
            logger.error(f"Validation error when deploying model {model_id}")
            raise
        except Exception as e:
            logger.error(f"Error deploying model {model_id}: {str(e)}")
            raise
    
    async def get_deployment_status(
        self,
        deployment_id: str
    ) -> Dict[str, Any]:
        """
        Get the status of a specific deployment.
        
        Args:
            deployment_id: The ID of the deployment
            
        Returns:
            Dictionary containing deployment status information
        """
        try:
            response = await self.client.get(f"/deployments/{deployment_id}")
            return response
        except ResourceNotFoundError:
            logger.error(f"Deployment with ID {deployment_id} not found")
            raise
        except Exception as e:
            logger.error(f"Error getting deployment status for {deployment_id}: {str(e)}")
            raise
    
    async def list_deployments(
        self,
        model_id: Optional[str] = None,
        deployment_target: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List deployments with optional filtering.
        
        Args:
            model_id: Filter by model ID
            deployment_target: Filter by deployment target
            status: Filter by deployment status
            
        Returns:
            List of dictionaries containing deployment information
        """
        params = {}
        if model_id:
            params["model_id"] = model_id
        if deployment_target:
            params["deployment_target"] = deployment_target
        if status:
            params["status"] = status
        
        try:
            response = await self.client.get("/deployments", params=params)
            return response.get("deployments", [])
        except Exception as e:
            logger.error(f"Error listing deployments: {str(e)}")
            raise
    
    async def undeploy_model(
        self,
        deployment_id: str
    ) -> bool:
        """
        Undeploy a model.
        
        Args:
            deployment_id: The ID of the deployment to remove
            
        Returns:
            True if the undeployment was successful, False otherwise
        """
        try:
            response = await self.client.delete(f"/deployments/{deployment_id}")
            return response.get("success", False)
        except ResourceNotFoundError:
            logger.error(f"Deployment with ID {deployment_id} not found")
            raise
        except Exception as e:
            logger.error(f"Error undeploying model with deployment ID {deployment_id}: {str(e)}")
            raise


class MLMetricsProviderAdapter(IMLMetricsProvider):
    """Adapter for ML Metrics Provider operations."""
    
    def __init__(self, client: Optional[AsyncHTTPServiceClient] = None, config: Optional[ServiceClientConfig] = None):
        """
        Initialize the adapter.
        
        Args:
            client: Optional pre-configured HTTP client
            config: Optional client configuration
        """
        self.client = client or AsyncHTTPServiceClient(
            config or ServiceClientConfig(
                base_url="http://ml-integration-service:8000/api/v1",
                timeout=30
            )
        )
    
    async def get_model_metrics(
        self,
        model_id: str,
        metric_types: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get metrics for a specific model.
        
        Args:
            model_id: The ID of the model
            metric_types: Optional list of metric types to retrieve
            start_time: Optional start time for metrics
            end_time: Optional end time for metrics
            
        Returns:
            Dictionary containing model metrics
        """
        params = {}
        if metric_types:
            params["metric_types"] = ",".join(metric_types)
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()
        
        try:
            response = await self.client.get(f"/models/{model_id}/metrics", params=params)
            return response
        except ResourceNotFoundError:
            logger.error(f"Model with ID {model_id} not found")
            raise
        except Exception as e:
            logger.error(f"Error getting metrics for model {model_id}: {str(e)}")
            raise
    
    async def record_model_metrics(
        self,
        model_id: str,
        metrics: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Record metrics for a specific model.
        
        Args:
            model_id: The ID of the model
            metrics: Dictionary of metrics to record
            timestamp: Optional timestamp for the metrics
            
        Returns:
            True if the metrics were recorded successfully, False otherwise
        """
        payload = {
            "metrics": metrics
        }
        
        if timestamp:
            payload["timestamp"] = timestamp.isoformat()
        
        try:
            response = await self.client.post(f"/models/{model_id}/metrics", json=payload)
            return response.get("success", False)
        except ResourceNotFoundError:
            logger.error(f"Model with ID {model_id} not found")
            raise
        except ValidationError:
            logger.error(f"Validation error when recording metrics for model {model_id}")
            raise
        except Exception as e:
            logger.error(f"Error recording metrics for model {model_id}: {str(e)}")
            raise
    
    async def get_service_metrics(
        self,
        metric_types: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get metrics for the ML Integration Service.
        
        Args:
            metric_types: Optional list of metric types to retrieve
            start_time: Optional start time for metrics
            end_time: Optional end time for metrics
            
        Returns:
            Dictionary containing service metrics
        """
        params = {}
        if metric_types:
            params["metric_types"] = ",".join(metric_types)
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()
        
        try:
            response = await self.client.get("/metrics", params=params)
            return response
        except Exception as e:
            logger.error(f"Error getting service metrics: {str(e)}")
            raise
