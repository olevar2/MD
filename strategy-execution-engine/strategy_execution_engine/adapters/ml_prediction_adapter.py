"""
ML Prediction Adapter Module

This module provides adapter implementations for ML prediction interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
import asyncio
import json
import os
import httpx
from common_lib.ml.prediction_interfaces import IMLPredictionService, IMLSignalGenerator, ModelType, ModelMetadata, PredictionRequest, PredictionResult
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from strategy_execution_engine.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class MLPredictionServiceAdapter(IMLPredictionService):
    """
    Adapter for ML prediction service that implements the common interface.
    
    This adapter can either use a direct API connection to the ML integration service
    or provide standalone functionality to avoid circular dependencies.
    """

    def __init__(self, config: Dict[str, Any]=None):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        ml_integration_base_url = self.config.get('ml_integration_base_url',
            os.environ.get('ML_INTEGRATION_BASE_URL',
            'http://ml-integration-service:8000'))
        self.client = httpx.AsyncClient(base_url=
            f"{ml_integration_base_url.rstrip('/')}/api/v1", timeout=30.0)
        self.metadata_cache = {}
        self.cache_ttl = self.config_manager.get('cache_ttl_minutes', 15)

    @async_with_exception_handling
    async def get_prediction(self, model_id: str, inputs: Dict[str, Any],
        version_id: Optional[str]=None, explanation_required: bool=False,
        context: Optional[Dict[str, Any]]=None) ->PredictionResult:
        """Get a prediction from a model."""
        try:
            request_data = {'inputs': inputs, 'explanation_required':
                explanation_required}
            if version_id:
                request_data['version_id'] = version_id
            if context:
                request_data['context'] = context
            response = await self.client.post(f'/models/{model_id}/predict',
                json=request_data)
            response.raise_for_status()
            result = response.json()
            return PredictionResult(prediction=result.get('prediction'),
                confidence=result.get('confidence', 0.0), model_id=model_id,
                version_id=result.get('version_id', version_id or 'default'
                ), timestamp=datetime.fromisoformat(result.get('timestamp')
                ) if 'timestamp' in result else datetime.now(), explanation
                =result.get('explanation'), metadata=result.get('metadata'))
        except Exception as e:
            logger.error(f'Error getting prediction: {str(e)}')
            return PredictionResult(prediction=None, confidence=0.0,
                model_id=model_id, version_id=version_id or 'default',
                timestamp=datetime.now(), explanation=None, metadata={
                'error': str(e), 'is_fallback': True})

    @async_with_exception_handling
    async def get_batch_predictions(self, model_id: str, batch_inputs: List
        [Dict[str, Any]], version_id: Optional[str]=None,
        explanation_required: bool=False, context: Optional[Dict[str, Any]]
        =None) ->List[PredictionResult]:
        """Get predictions for a batch of inputs."""
        try:
            request_data = {'batch_inputs': batch_inputs,
                'explanation_required': explanation_required}
            if version_id:
                request_data['version_id'] = version_id
            if context:
                request_data['context'] = context
            response = await self.client.post(
                f'/models/{model_id}/batch-predict', json=request_data)
            response.raise_for_status()
            results = response.json()
            return [PredictionResult(prediction=result.get('prediction'),
                confidence=result.get('confidence', 0.0), model_id=model_id,
                version_id=result.get('version_id', version_id or 'default'
                ), timestamp=datetime.fromisoformat(result.get('timestamp')
                ) if 'timestamp' in result else datetime.now(), explanation
                =result.get('explanation'), metadata=result.get('metadata')
                ) for result in results]
        except Exception as e:
            logger.error(f'Error getting batch predictions: {str(e)}')
            return [PredictionResult(prediction=None, confidence=0.0,
                model_id=model_id, version_id=version_id or 'default',
                timestamp=datetime.now(), explanation=None, metadata={
                'error': str(e), 'is_fallback': True}) for _ in batch_inputs]

    @async_with_exception_handling
    async def get_model_metadata(self, model_id: str, version_id: Optional[
        str]=None) ->ModelMetadata:
        """Get metadata for a model."""
        try:
            cache_key = f"{model_id}_{version_id or 'default'}"
            if cache_key in self.metadata_cache:
                cache_entry = self.metadata_cache[cache_key]
                cache_age = (datetime.now() - cache_entry['timestamp']
                    ).total_seconds() / 60
                if cache_age < self.cache_ttl:
                    return cache_entry['metadata']
            params = {}
            if version_id:
                params['version_id'] = version_id
            response = await self.client.get(f'/models/{model_id}/metadata',
                params=params)
            response.raise_for_status()
            result = response.json()
            metadata = ModelMetadata(model_id=model_id, model_type=
                ModelType(result.get('model_type', 'custom')), version=
                result.get('version', version_id or 'default'), created_at=
                datetime.fromisoformat(result.get('created_at')) if 
                'created_at' in result else datetime.now(), features=result
                .get('features', []), target=result.get('target', ''),
                description=result.get('description'), performance_metrics=
                result.get('performance_metrics'), tags=result.get('tags'))
            self.metadata_cache[cache_key] = {'metadata': metadata,
                'timestamp': datetime.now()}
            return metadata
        except Exception as e:
            logger.error(f'Error getting model metadata: {str(e)}')
            return ModelMetadata(model_id=model_id, model_type=ModelType.
                CUSTOM, version=version_id or 'default', created_at=
                datetime.now(), features=[], target='', description=
                f'Error: {str(e)}', performance_metrics=None, tags=['error',
                'fallback'])

    @async_with_exception_handling
    async def list_available_models(self, model_type: Optional[ModelType]=
        None, tags: Optional[List[str]]=None) ->List[ModelMetadata]:
        """List available models."""
        try:
            params = {}
            if model_type:
                params['model_type'] = model_type.value
            if tags:
                params['tags'] = ','.join(tags)
            response = await self.client.get('/models', params=params)
            response.raise_for_status()
            results = response.json()
            return [ModelMetadata(model_id=result.get('model_id'),
                model_type=ModelType(result.get('model_type', 'custom')),
                version=result.get('version', 'default'), created_at=
                datetime.fromisoformat(result.get('created_at')) if 
                'created_at' in result else datetime.now(), features=result
                .get('features', []), target=result.get('target', ''),
                description=result.get('description'), performance_metrics=
                result.get('performance_metrics'), tags=result.get('tags')) for
                result in results]
        except Exception as e:
            logger.error(f'Error listing available models: {str(e)}')
            return []


class MLSignalGeneratorAdapter(IMLSignalGenerator):
    """
    Adapter for ML signal generation that implements the common interface.
    
    This adapter can either use a direct API connection to the ML integration service
    or provide standalone functionality to avoid circular dependencies.
    """

    def __init__(self, config: Dict[str, Any]=None):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        ml_integration_base_url = self.config.get('ml_integration_base_url',
            os.environ.get('ML_INTEGRATION_BASE_URL',
            'http://ml-integration-service:8000'))
        self.client = httpx.AsyncClient(base_url=
            f"{ml_integration_base_url.rstrip('/')}/api/v1", timeout=30.0)

    @async_with_exception_handling
    async def generate_trading_signals(self, symbol: str, timeframe: str,
        lookback_bars: int=100, models: Optional[List[str]]=None, context:
        Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
        """Generate trading signals using ML models."""
        try:
            request_data = {'symbol': symbol, 'timeframe': timeframe,
                'lookback_bars': lookback_bars}
            if models:
                request_data['models'] = models
            if context:
                request_data['context'] = context
            response = await self.client.post('/signals/generate', json=
                request_data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f'Error generating trading signals: {str(e)}')
            return {'symbol': symbol, 'timeframe': timeframe, 'signals': [],
                'timestamp': datetime.now().isoformat(), 'error': str(e),
                'is_fallback': True}

    @async_with_exception_handling
    async def get_model_performance(self, model_id: str, symbol: Optional[
        str]=None, timeframe: Optional[str]=None, start_date: Optional[
        datetime]=None, end_date: Optional[datetime]=None) ->Dict[str, Any]:
        """Get performance metrics for a model."""
        try:
            params = {'model_id': model_id}
            if symbol:
                params['symbol'] = symbol
            if timeframe:
                params['timeframe'] = timeframe
            if start_date:
                params['start_date'] = start_date.isoformat()
            if end_date:
                params['end_date'] = end_date.isoformat()
            response = await self.client.get('/models/performance', params=
                params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f'Error getting model performance: {str(e)}')
            return {'model_id': model_id, 'symbol': symbol, 'timeframe':
                timeframe, 'metrics': {}, 'error': str(e), 'is_fallback': True}
