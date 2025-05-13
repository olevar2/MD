"""
ML Workbench Client

This module provides a client for interacting with the ML Workbench Service.
It uses the standardized client implementation from common-lib.
"""
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from common_lib.clients import BaseServiceClient, ClientConfig
from common_lib.clients.exceptions import ClientError
logger = logging.getLogger(__name__)


from ml_integration_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class MLWorkbenchClient(BaseServiceClient):
    """
    Client for interacting with the ML Workbench Service.
    
    This client provides methods for:
    1. Managing ML models
    2. Training and evaluating models
    3. Serving model predictions
    4. Monitoring model performance
    """

    def __init__(self, config: Union[ClientConfig, Dict[str, Any]]):
        """
        Initialize the ML Workbench client.
        
        Args:
            config: Client configuration
        """
        super().__init__(config)
        logger.info(
            f'ML Workbench Client initialized with base URL: {self.base_url}')

    @async_with_exception_handling
    async def get_models(self, model_type: Optional[str]=None, status:
        Optional[str]=None, limit: int=100, offset: int=0) ->List[Dict[str,
        Any]]:
        """
        Get a list of ML models.
        
        Args:
            model_type: Filter by model type
            status: Filter by model status
            limit: Maximum number of models to return
            offset: Offset for pagination
            
        Returns:
            List of model information dictionaries
            
        Raises:
            ClientError: If the request fails
        """
        params = {'limit': limit, 'offset': offset}
        if model_type:
            params['model_type'] = model_type
        if status:
            params['status'] = status
        try:
            response = await self.get('models', params=params)
            return response.get('models', [])
        except Exception as e:
            logger.error(f'Error getting models: {str(e)}')
            raise ClientError('Failed to get models', service_name=self.
                config.service_name) from e

    @async_with_exception_handling
    async def get_model(self, model_id: str) ->Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model information dictionary
            
        Raises:
            ClientError: If the request fails
        """
        try:
            return await self.get(f'models/{model_id}')
        except Exception as e:
            logger.error(f'Error getting model {model_id}: {str(e)}')
            raise ClientError(f'Failed to get model {model_id}',
                service_name=self.config.service_name) from e

    @async_with_exception_handling
    async def create_model(self, name: str, model_type: str, description:
        Optional[str]=None, metadata: Optional[Dict[str, Any]]=None) ->Dict[
        str, Any]:
        """
        Create a new ML model.
        
        Args:
            name: Model name
            model_type: Model type
            description: Model description
            metadata: Additional metadata
            
        Returns:
            Created model information
            
        Raises:
            ClientError: If the request fails
        """
        data = {'name': name, 'model_type': model_type}
        if description:
            data['description'] = description
        if metadata:
            data['metadata'] = metadata
        try:
            return await self.post('models', data=data)
        except Exception as e:
            logger.error(f'Error creating model: {str(e)}')
            raise ClientError('Failed to create model', service_name=self.
                config.service_name) from e

    @async_with_exception_handling
    async def start_training(self, model_id: str, training_config: Dict[str,
        Any], dataset_id: Optional[str]=None) ->Dict[str, Any]:
        """
        Start training a model.
        
        Args:
            model_id: Model ID
            training_config: Training configuration
            dataset_id: Dataset ID
            
        Returns:
            Training job information
            
        Raises:
            ClientError: If the request fails
        """
        data = {'training_config': training_config}
        if dataset_id:
            data['dataset_id'] = dataset_id
        try:
            return await self.post(f'models/{model_id}/train', data=data)
        except Exception as e:
            logger.error(
                f'Error starting training for model {model_id}: {str(e)}')
            raise ClientError(f'Failed to start training for model {model_id}',
                service_name=self.config.service_name) from e

    @async_with_exception_handling
    async def get_training_status(self, model_id: str, job_id: str) ->Dict[
        str, Any]:
        """
        Get the status of a training job.
        
        Args:
            model_id: Model ID
            job_id: Training job ID
            
        Returns:
            Training job status
            
        Raises:
            ClientError: If the request fails
        """
        try:
            return await self.get(f'models/{model_id}/train/{job_id}')
        except Exception as e:
            logger.error(
                f'Error getting training status for job {job_id}: {str(e)}')
            raise ClientError(f'Failed to get training status for job {job_id}'
                , service_name=self.config.service_name) from e

    @async_with_exception_handling
    async def predict(self, model_id: str, inputs: Dict[str, Any],
        version_id: Optional[str]=None) ->Dict[str, Any]:
        """
        Get predictions from a model.
        
        Args:
            model_id: Model ID
            inputs: Input data
            version_id: Model version ID
            
        Returns:
            Prediction results
            
        Raises:
            ClientError: If the request fails
        """
        data = {'inputs': inputs}
        if version_id:
            data['version_id'] = version_id
        try:
            return await self.post(f'models/{model_id}/predict', data=data)
        except Exception as e:
            logger.error(
                f'Error getting predictions from model {model_id}: {str(e)}')
            raise ClientError(
                f'Failed to get predictions from model {model_id}',
                service_name=self.config.service_name) from e

    @async_with_exception_handling
    async def get_model_metrics(self, model_id: str, start_time: Optional[
        Union[str, datetime]]=None, end_time: Optional[Union[str, datetime]
        ]=None, metric_types: Optional[List[str]]=None) ->Dict[str, Any]:
        """
        Get performance metrics for a model.
        
        Args:
            model_id: Model ID
            start_time: Start time for metrics
            end_time: End time for metrics
            metric_types: Types of metrics to retrieve
            
        Returns:
            Model metrics
            
        Raises:
            ClientError: If the request fails
        """
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()
        params = {}
        if start_time:
            params['start_time'] = start_time
        if end_time:
            params['end_time'] = end_time
        if metric_types:
            params['metric_types'] = ','.join(metric_types)
        try:
            return await self.get(f'models/{model_id}/metrics', params=params)
        except Exception as e:
            logger.error(
                f'Error getting metrics for model {model_id}: {str(e)}')
            raise ClientError(f'Failed to get metrics for model {model_id}',
                service_name=self.config.service_name) from e
