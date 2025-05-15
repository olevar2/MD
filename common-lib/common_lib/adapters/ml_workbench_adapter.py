"""
ML Workbench Service Adapter.

This module provides adapter implementations for the ML Workbench Service interfaces.
These adapters allow other services to interact with the ML Workbench Service
without direct dependencies, breaking circular dependencies.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

import pandas as pd

from common_lib.interfaces.ml_workbench import IExperimentManager, IModelEvaluator, IDatasetManager
from common_lib.service_client.base_client import ServiceClientConfig
from common_lib.service_client.http_client import AsyncHTTPServiceClient
from common_lib.errors.base_exceptions import ServiceError, ValidationError
from common_lib.resilience.decorators import with_circuit_breaker, with_retry, with_timeout


logger = logging.getLogger(__name__)


class MLWorkbenchAdapter:
    """Adapter for the ML Workbench Service."""
    
    def __init__(self, config: Union[ServiceClientConfig, Dict[str, Any]]):
        """
        Initialize the adapter with a service client configuration.
        
        Args:
            config: Service client configuration
        """
        if isinstance(config, dict):
            config = ServiceClientConfig(**config)
        
        self.client = AsyncHTTPServiceClient(config)
        self.logger = logger
        
        # Initialize sub-adapters
        self.experiment_manager = ExperimentManagerAdapter(self.client)
        self.model_evaluator = ModelEvaluatorAdapter(self.client)
        self.dataset_manager = DatasetManagerAdapter(self.client)


class ExperimentManagerAdapter(IExperimentManager):
    """Adapter implementation for the Experiment Manager."""
    
    def __init__(self, client: AsyncHTTPServiceClient):
        """
        Initialize the adapter with a service client.
        
        Args:
            client: HTTP client for API calls
        """
        self.client = client
        self.logger = logger
    
    @with_circuit_breaker("ml_workbench.create_experiment")
    @with_retry(max_retries=3, retry_delay=1.0)
    @with_timeout(timeout_seconds=10.0)
    async def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new experiment.
        
        Args:
            name: Name of the experiment
            description: Optional description
            tags: Optional tags
            metadata: Optional metadata
            
        Returns:
            Dictionary with experiment details
        """
        try:
            payload = {
                "name": name
            }
            
            if description:
                payload["description"] = description
            if tags:
                payload["tags"] = tags
            if metadata:
                payload["metadata"] = metadata
            
            response = await self.client.post("/experiments", json=payload)
            return response
        except Exception as e:
            self.logger.error(f"Error creating experiment: {str(e)}")
            raise ServiceError(f"Failed to create experiment: {str(e)}")
    
    @with_circuit_breaker("ml_workbench.get_experiment")
    @with_retry(max_retries=3, retry_delay=1.0)
    @with_timeout(timeout_seconds=5.0)
    async def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get details of an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Dictionary with experiment details
        """
        try:
            response = await self.client.get(f"/experiments/{experiment_id}")
            return response
        except Exception as e:
            self.logger.error(f"Error getting experiment: {str(e)}")
            raise ServiceError(f"Failed to get experiment: {str(e)}")
    
    @with_circuit_breaker("ml_workbench.list_experiments")
    @with_retry(max_retries=3, retry_delay=1.0)
    @with_timeout(timeout_seconds=5.0)
    async def list_experiments(
        self,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List experiments.
        
        Args:
            tags: Optional tags to filter by
            status: Optional status to filter by
            limit: Maximum number of results
            offset: Result offset
            
        Returns:
            Dictionary with experiments and pagination information
        """
        try:
            params = {
                "limit": limit,
                "offset": offset
            }
            
            if tags:
                params["tags"] = ",".join(tags)
            if status:
                params["status"] = status
            
            response = await self.client.get("/experiments", params=params)
            return response
        except Exception as e:
            self.logger.error(f"Error listing experiments: {str(e)}")
            raise ServiceError(f"Failed to list experiments: {str(e)}")
    
    @with_circuit_breaker("ml_workbench.update_experiment")
    @with_retry(max_retries=3, retry_delay=1.0)
    @with_timeout(timeout_seconds=10.0)
    async def update_experiment(
        self,
        experiment_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update an experiment.
        
        Args:
            experiment_id: ID of the experiment
            updates: Dictionary of updates to apply
            
        Returns:
            Dictionary with updated experiment details
        """
        try:
            response = await self.client.patch(f"/experiments/{experiment_id}", json=updates)
            return response
        except Exception as e:
            self.logger.error(f"Error updating experiment: {str(e)}")
            raise ServiceError(f"Failed to update experiment: {str(e)}")
    
    @with_circuit_breaker("ml_workbench.delete_experiment")
    @with_retry(max_retries=3, retry_delay=1.0)
    @with_timeout(timeout_seconds=10.0)
    async def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            True if the experiment was deleted successfully
        """
        try:
            response = await self.client.delete(f"/experiments/{experiment_id}")
            return response.get("success", False)
        except Exception as e:
            self.logger.error(f"Error deleting experiment: {str(e)}")
            raise ServiceError(f"Failed to delete experiment: {str(e)}")


class ModelEvaluatorAdapter(IModelEvaluator):
    """Adapter implementation for the Model Evaluator."""
    
    def __init__(self, client: AsyncHTTPServiceClient):
        """
        Initialize the adapter with a service client.
        
        Args:
            client: HTTP client for API calls
        """
        self.client = client
        self.logger = logger
    
    @with_circuit_breaker("ml_workbench.evaluate_model")
    @with_retry(max_retries=3, retry_delay=1.0)
    @with_timeout(timeout_seconds=30.0)
    async def evaluate_model(
        self,
        model_id: str,
        dataset_id: str,
        metrics: List[str],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a model on a dataset.
        
        Args:
            model_id: ID of the model
            dataset_id: ID of the dataset
            metrics: List of metrics to calculate
            parameters: Optional evaluation parameters
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            payload = {
                "model_id": model_id,
                "dataset_id": dataset_id,
                "metrics": metrics
            }
            
            if parameters:
                payload["parameters"] = parameters
            
            response = await self.client.post("/evaluations", json=payload)
            return response
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise ServiceError(f"Failed to evaluate model: {str(e)}")
    
    @with_circuit_breaker("ml_workbench.compare_models")
    @with_retry(max_retries=3, retry_delay=1.0)
    @with_timeout(timeout_seconds=30.0)
    async def compare_models(
        self,
        model_ids: List[str],
        dataset_id: str,
        metrics: List[str],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models on a dataset.
        
        Args:
            model_ids: IDs of the models
            dataset_id: ID of the dataset
            metrics: List of metrics to calculate
            parameters: Optional comparison parameters
            
        Returns:
            Dictionary with comparison results
        """
        try:
            payload = {
                "model_ids": model_ids,
                "dataset_id": dataset_id,
                "metrics": metrics
            }
            
            if parameters:
                payload["parameters"] = parameters
            
            response = await self.client.post("/comparisons", json=payload)
            return response
        except Exception as e:
            self.logger.error(f"Error comparing models: {str(e)}")
            raise ServiceError(f"Failed to compare models: {str(e)}")
    
    @with_circuit_breaker("ml_workbench.get_evaluation_history")
    @with_retry(max_retries=3, retry_delay=1.0)
    @with_timeout(timeout_seconds=5.0)
    async def get_evaluation_history(
        self,
        model_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get evaluation history for a model.
        
        Args:
            model_id: ID of the model
            limit: Maximum number of results
            
        Returns:
            List of evaluation results
        """
        try:
            params = {
                "model_id": model_id,
                "limit": limit
            }
            
            response = await self.client.get("/evaluations/history", params=params)
            return response.get("evaluations", [])
        except Exception as e:
            self.logger.error(f"Error getting evaluation history: {str(e)}")
            raise ServiceError(f"Failed to get evaluation history: {str(e)}")


class DatasetManagerAdapter(IDatasetManager):
    """Adapter implementation for the Dataset Manager."""
    
    def __init__(self, client: AsyncHTTPServiceClient):
        """
        Initialize the adapter with a service client.
        
        Args:
            client: HTTP client for API calls
        """
        self.client = client
        self.logger = logger
    
    @with_circuit_breaker("ml_workbench.create_dataset")
    @with_retry(max_retries=3, retry_delay=1.0)
    @with_timeout(timeout_seconds=30.0)
    async def create_dataset(
        self,
        name: str,
        data: Union[pd.DataFrame, Dict[str, Any]],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new dataset.
        
        Args:
            name: Name of the dataset
            data: Dataset data
            description: Optional description
            tags: Optional tags
            metadata: Optional metadata
            
        Returns:
            Dictionary with dataset details
        """
        try:
            # Convert DataFrame to dict if needed
            if isinstance(data, pd.DataFrame):
                data_dict = data.to_dict(orient="records")
            else:
                data_dict = data
            
            payload = {
                "name": name,
                "data": data_dict
            }
            
            if description:
                payload["description"] = description
            if tags:
                payload["tags"] = tags
            if metadata:
                payload["metadata"] = metadata
            
            response = await self.client.post("/datasets", json=payload)
            return response
        except Exception as e:
            self.logger.error(f"Error creating dataset: {str(e)}")
            raise ServiceError(f"Failed to create dataset: {str(e)}")
    
    @with_circuit_breaker("ml_workbench.get_dataset")
    @with_retry(max_retries=3, retry_delay=1.0)
    @with_timeout(timeout_seconds=10.0)
    async def get_dataset(
        self,
        dataset_id: str,
        include_data: bool = False
    ) -> Dict[str, Any]:
        """
        Get details of a dataset.
        
        Args:
            dataset_id: ID of the dataset
            include_data: Whether to include the dataset data
            
        Returns:
            Dictionary with dataset details
        """
        try:
            params = {
                "include_data": str(include_data).lower()
            }
            
            response = await self.client.get(f"/datasets/{dataset_id}", params=params)
            return response
        except Exception as e:
            self.logger.error(f"Error getting dataset: {str(e)}")
            raise ServiceError(f"Failed to get dataset: {str(e)}")
    
    @with_circuit_breaker("ml_workbench.list_datasets")
    @with_retry(max_retries=3, retry_delay=1.0)
    @with_timeout(timeout_seconds=5.0)
    async def list_datasets(
        self,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List datasets.
        
        Args:
            tags: Optional tags to filter by
            limit: Maximum number of results
            offset: Result offset
            
        Returns:
            Dictionary with datasets and pagination information
        """
        try:
            params = {
                "limit": limit,
                "offset": offset
            }
            
            if tags:
                params["tags"] = ",".join(tags)
            
            response = await self.client.get("/datasets", params=params)
            return response
        except Exception as e:
            self.logger.error(f"Error listing datasets: {str(e)}")
            raise ServiceError(f"Failed to list datasets: {str(e)}")
    
    @with_circuit_breaker("ml_workbench.update_dataset")
    @with_retry(max_retries=3, retry_delay=1.0)
    @with_timeout(timeout_seconds=10.0)
    async def update_dataset(
        self,
        dataset_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update a dataset.
        
        Args:
            dataset_id: ID of the dataset
            updates: Dictionary of updates to apply
            
        Returns:
            Dictionary with updated dataset details
        """
        try:
            response = await self.client.patch(f"/datasets/{dataset_id}", json=updates)
            return response
        except Exception as e:
            self.logger.error(f"Error updating dataset: {str(e)}")
            raise ServiceError(f"Failed to update dataset: {str(e)}")
    
    @with_circuit_breaker("ml_workbench.delete_dataset")
    @with_retry(max_retries=3, retry_delay=1.0)
    @with_timeout(timeout_seconds=10.0)
    async def delete_dataset(self, dataset_id: str) -> bool:
        """
        Delete a dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            True if the dataset was deleted successfully
        """
        try:
            response = await self.client.delete(f"/datasets/{dataset_id}")
            return response.get("success", False)
        except Exception as e:
            self.logger.error(f"Error deleting dataset: {str(e)}")
            raise ServiceError(f"Failed to delete dataset: {str(e)}")
