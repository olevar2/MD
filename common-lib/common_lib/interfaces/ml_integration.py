"""
ML Integration Service Interfaces

This module defines interfaces for the ML Integration Service, allowing other services
to interact with it without direct dependencies.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


class ModelStatus(str, Enum):
    """Status of a machine learning model."""
    TRAINING = "training"
    READY = "ready"
    FAILED = "failed"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ModelType(str, Enum):
    """Types of machine learning models."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    REINFORCEMENT = "reinforcement"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    CUSTOM = "custom"


class ModelFramework(str, Enum):
    """Machine learning frameworks."""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    SCIKIT_LEARN = "scikit_learn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CUSTOM = "custom"


class IMLModelRegistry(ABC):
    """Interface for ML model registry operations."""
    
    @abstractmethod
    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_id: The ID of the model
            
        Returns:
            Dictionary containing model information
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass


class IMLJobTracker(ABC):
    """Interface for ML job tracking operations."""
    
    @abstractmethod
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a specific job.
        
        Args:
            job_id: The ID of the job
            
        Returns:
            Dictionary containing job status information
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass


class IMLModelDeployment(ABC):
    """Interface for ML model deployment operations."""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass


class IMLMetricsProvider(ABC):
    """Interface for ML metrics operations."""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
