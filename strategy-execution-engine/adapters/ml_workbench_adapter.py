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


from core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
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
