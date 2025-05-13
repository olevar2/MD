"""
ML Workbench Adapter Module

This module provides adapter implementations for ML workbench interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
import asyncio
import json
import os
import httpx
from common_lib.ml.workbench_interfaces import IModelOptimizationService, IModelRegistryService, IReinforcementLearningService, ModelOptimizationType, OptimizationConfig, OptimizationResult
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class ModelOptimizationServiceAdapter(IModelOptimizationService):
    """
    Adapter for model optimization service that implements the common interface.
    
    This adapter can either use a direct API connection to the ML workbench service
    or provide standalone functionality to avoid circular dependencies.
    """

    def __init__(self, config: Dict[str, Any]=None):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        ml_workbench_base_url = self.config.get('ml_workbench_base_url', os
            .environ.get('ML_WORKBENCH_BASE_URL',
            'http://ml_workbench-service:8000'))
        self.client = httpx.AsyncClient(base_url=
            f"{ml_workbench_base_url.rstrip('/')}/api/v1", timeout=30.0)

    @async_with_exception_handling
    async def optimize_model(self, config: OptimizationConfig) ->str:
        """Start a model optimization job."""
        try:
            request_data = {'model_id': config.model_id,
                'optimization_type': config.optimization_type.value,
                'parameters': config.parameters, 'max_iterations': config.
                max_iterations, 'target_metric': config.target_metric}
            if config.constraints:
                request_data['constraints'] = config.constraints
            if config.timeout_minutes:
                request_data['timeout_minutes'] = config.timeout_minutes
            response = await self.client.post('/optimization/start', json=
                request_data)
            response.raise_for_status()
            result = response.json()
            return result.get('optimization_id', '')
        except Exception as e:
            logger.error(f'Error starting model optimization: {str(e)}')
            return ''

    @with_resilience('get_optimization_status')
    @async_with_exception_handling
    async def get_optimization_status(self, optimization_id: str) ->Dict[
        str, Any]:
        """Get the status of an optimization job."""
        try:
            response = await self.client.get(
                f'/optimization/{optimization_id}/status')
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f'Error getting optimization status: {str(e)}')
            return {'optimization_id': optimization_id, 'status': 'error',
                'error': str(e), 'progress': 0.0, 'is_fallback': True}

    @with_resilience('get_optimization_result')
    @async_with_exception_handling
    async def get_optimization_result(self, optimization_id: str
        ) ->OptimizationResult:
        """Get the result of an optimization job."""
        try:
            response = await self.client.get(
                f'/optimization/{optimization_id}/result')
            response.raise_for_status()
            result = response.json()
            return OptimizationResult(model_id=result.get('model_id', ''),
                optimization_id=optimization_id, best_parameters=result.get
                ('best_parameters', {}), best_score=result.get('best_score',
                0.0), iterations_completed=result.get(
                'iterations_completed', 0), timestamp=datetime.
                fromisoformat(result.get('timestamp')) if 'timestamp' in
                result else datetime.now(), history=result.get('history'),
                metadata=result.get('metadata'))
        except Exception as e:
            logger.error(f'Error getting optimization result: {str(e)}')
            return OptimizationResult(model_id='', optimization_id=
                optimization_id, best_parameters={}, best_score=0.0,
                iterations_completed=0, timestamp=datetime.now(), history=
                None, metadata={'error': str(e), 'is_fallback': True})

    @async_with_exception_handling
    async def cancel_optimization(self, optimization_id: str) ->bool:
        """Cancel an optimization job."""
        try:
            response = await self.client.post(
                f'/optimization/{optimization_id}/cancel')
            response.raise_for_status()
            result = response.json()
            return result.get('success', False)
        except Exception as e:
            logger.error(f'Error canceling optimization: {str(e)}')
            return False


class ModelRegistryServiceAdapter(IModelRegistryService):
    """
    Adapter for model registry service that implements the common interface.
    
    This adapter can either use a direct API connection to the ML workbench service
    or provide standalone functionality to avoid circular dependencies.
    """

    def __init__(self, config: Dict[str, Any]=None):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        ml_workbench_base_url = self.config.get('ml_workbench_base_url', os
            .environ.get('ML_WORKBENCH_BASE_URL',
            'http://ml_workbench-service:8000'))
        self.client = httpx.AsyncClient(base_url=
            f"{ml_workbench_base_url.rstrip('/')}/api/v1", timeout=30.0)
        self.model_info_cache = {}
        self.cache_ttl = self.config_manager.get('cache_ttl_minutes', 15)

    @async_with_exception_handling
    async def register_model(self, model_id: str, model_type: str, version:
        str, metadata: Dict[str, Any], artifacts_path: str) ->Dict[str, Any]:
        """Register a model in the registry."""
        try:
            request_data = {'model_id': model_id, 'model_type': model_type,
                'version': version, 'metadata': metadata, 'artifacts_path':
                artifacts_path}
            response = await self.client.post('/models/register', json=
                request_data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f'Error registering model: {str(e)}')
            return {'model_id': model_id, 'version': version, 'status':
                'error', 'error': str(e), 'is_fallback': True}

    @with_resilience('get_model_info')
    @async_with_exception_handling
    async def get_model_info(self, model_id: str, version: Optional[str]=None
        ) ->Dict[str, Any]:
        """Get information about a model."""
        try:
            cache_key = f"{model_id}_{version or 'latest'}"
            if cache_key in self.model_info_cache:
                cache_entry = self.model_info_cache[cache_key]
                cache_age = (datetime.now() - cache_entry['timestamp']
                    ).total_seconds() / 60
                if cache_age < self.cache_ttl:
                    return cache_entry['info']
            params = {}
            if version:
                params['version'] = version
            response = await self.client.get(f'/models/{model_id}', params=
                params)
            response.raise_for_status()
            info = response.json()
            self.model_info_cache[cache_key] = {'info': info, 'timestamp':
                datetime.now()}
            return info
        except Exception as e:
            logger.error(f'Error getting model info: {str(e)}')
            return {'model_id': model_id, 'version': version or 'latest',
                'status': 'error', 'error': str(e), 'is_fallback': True}

    @async_with_exception_handling
    async def list_models(self, model_type: Optional[str]=None, tags:
        Optional[List[str]]=None, limit: int=100, offset: int=0) ->List[Dict
        [str, Any]]:
        """List models in the registry."""
        try:
            params = {'limit': limit, 'offset': offset}
            if model_type:
                params['model_type'] = model_type
            if tags:
                params['tags'] = ','.join(tags)
            response = await self.client.get('/models', params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f'Error listing models: {str(e)}')
            return []

    @with_resilience('delete_model')
    @async_with_exception_handling
    async def delete_model(self, model_id: str, version: Optional[str]=None
        ) ->bool:
        """Delete a model from the registry."""
        try:
            params = {}
            if version:
                params['version'] = version
            response = await self.client.delete(f'/models/{model_id}',
                params=params)
            response.raise_for_status()
            result = response.json()
            cache_keys_to_remove = [k for k in self.model_info_cache.keys() if
                k.startswith(f'{model_id}_')]
            for key in cache_keys_to_remove:
                self.model_info_cache.pop(key, None)
            return result.get('success', False)
        except Exception as e:
            logger.error(f'Error deleting model: {str(e)}')
            return False


class ReinforcementLearningServiceAdapter(IReinforcementLearningService):
    """
    Adapter for reinforcement learning service that implements the common interface.
    
    This adapter can either use a direct API connection to the ML workbench service
    or provide standalone functionality to avoid circular dependencies.
    """

    def __init__(self, config: Dict[str, Any]=None):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        ml_workbench_base_url = self.config.get('ml_workbench_base_url', os
            .environ.get('ML_WORKBENCH_BASE_URL',
            'http://ml_workbench-service:8000'))
        self.client = httpx.AsyncClient(base_url=
            f"{ml_workbench_base_url.rstrip('/')}/api/v1", timeout=30.0)

    @async_with_exception_handling
    async def train_rl_model(self, model_id: str, environment_config: Dict[
        str, Any], agent_config: Dict[str, Any], training_config: Dict[str,
        Any]) ->str:
        """Train a reinforcement learning model."""
        try:
            request_data = {'model_id': model_id, 'environment_config':
                environment_config, 'agent_config': agent_config,
                'training_config': training_config}
            response = await self.client.post('/rl/train', json=request_data)
            response.raise_for_status()
            result = response.json()
            return result.get('training_id', '')
        except Exception as e:
            logger.error(f'Error training RL model: {str(e)}')
            return ''

    @with_resilience('get_training_status')
    @async_with_exception_handling
    async def get_training_status(self, training_id: str) ->Dict[str, Any]:
        """Get the status of a training job."""
        try:
            response = await self.client.get(
                f'/rl/training/{training_id}/status')
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f'Error getting training status: {str(e)}')
            return {'training_id': training_id, 'status': 'error', 'error':
                str(e), 'progress': 0.0, 'is_fallback': True}

    @with_resilience('get_rl_model_performance')
    @async_with_exception_handling
    async def get_rl_model_performance(self, model_id: str,
        environment_config: Dict[str, Any]) ->Dict[str, Any]:
        """Get performance metrics for a reinforcement learning model."""
        try:
            request_data = {'environment_config': environment_config}
            response = await self.client.post(
                f'/rl/models/{model_id}/performance', json=request_data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f'Error getting RL model performance: {str(e)}')
            return {'model_id': model_id, 'metrics': {}, 'error': str(e),
                'is_fallback': True}

    @with_resilience('get_rl_model_action')
    @async_with_exception_handling
    async def get_rl_model_action(self, model_id: str, state: Dict[str, Any
        ], version: Optional[str]=None) ->Dict[str, Any]:
        """Get an action from a reinforcement learning model."""
        try:
            request_data = {'state': state}
            params = {}
            if version:
                params['version'] = version
            response = await self.client.post(f'/rl/models/{model_id}/action',
                params=params, json=request_data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f'Error getting RL model action: {str(e)}')
            return {'model_id': model_id, 'action': None, 'confidence': 0.0,
                'error': str(e), 'is_fallback': True}
