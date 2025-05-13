"""
Prediction Service Adapter Module

This module provides adapter implementations for prediction service interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
import asyncio
import json
import os
from common_lib.ml.prediction_interfaces import IMLPredictionService, IMLSignalGenerator, ModelType, ModelMetadata, PredictionRequest, PredictionResult
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from ml_integration_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class PredictionServiceAdapter(IMLPredictionService):
    """
    Adapter for prediction service that implements the common interface.
    
    This adapter provides direct implementation of the prediction service interface
    to avoid circular dependencies.
    """

    def __init__(self, model_service=None, config: Dict[str, Any]=None):
        """
        Initialize the adapter.
        
        Args:
            model_service: Optional model service instance to wrap
            config: Configuration parameters
        """
        self.config = config or {}
        self.model_service = model_service
        self.metadata_cache = {}
        self.cache_ttl = self.config_manager.get('cache_ttl_minutes', 15)

    @async_with_exception_handling
    async def get_prediction(self, model_id: str, inputs: Dict[str, Any],
        version_id: Optional[str]=None, explanation_required: bool=False,
        context: Optional[Dict[str, Any]]=None) ->PredictionResult:
        """Get a prediction from a model."""
        try:
            if self.model_service:
                result = await self.model_service.get_prediction(model_id=
                    model_id, inputs=inputs, version_id=version_id,
                    explanation_required=explanation_required, context=context)
                return result
            from ml_integration_service.services.prediction_service import get_prediction
            result = await get_prediction(model_id=model_id, inputs=inputs,
                version_id=version_id, explanation_required=
                explanation_required, context=context)
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
            if self.model_service:
                results = await self.model_service.get_batch_predictions(
                    model_id=model_id, batch_inputs=batch_inputs,
                    version_id=version_id, explanation_required=
                    explanation_required, context=context)
                return results
            from ml_integration_service.services.prediction_service import get_batch_predictions
            results = await get_batch_predictions(model_id=model_id,
                batch_inputs=batch_inputs, version_id=version_id,
                explanation_required=explanation_required, context=context)
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
            if self.model_service:
                metadata = await self.model_service.get_model_metadata(model_id
                    =model_id, version_id=version_id)
                self.metadata_cache[cache_key] = {'metadata': metadata,
                    'timestamp': datetime.now()}
                return metadata
            from ml_integration_service.services.model_registry import get_model_metadata
            result = await get_model_metadata(model_id=model_id, version_id
                =version_id)
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
            if self.model_service:
                return await self.model_service.list_available_models(
                    model_type=model_type, tags=tags)
            from ml_integration_service.services.model_registry import list_models
            results = await list_models(model_type=model_type.value if
                model_type else None, tags=tags)
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


class SignalGeneratorAdapter(IMLSignalGenerator):
    """
    Adapter for ML signal generation that implements the common interface.
    
    This adapter provides direct implementation of the signal generator interface
    to avoid circular dependencies.
    """

    def __init__(self, signal_service=None, config: Dict[str, Any]=None):
        """
        Initialize the adapter.
        
        Args:
            signal_service: Optional signal service instance to wrap
            config: Configuration parameters
        """
        self.config = config or {}
        self.signal_service = signal_service

    @async_with_exception_handling
    async def generate_trading_signals(self, symbol: str, timeframe: str,
        lookback_bars: int=100, models: Optional[List[str]]=None, context:
        Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
        """Generate trading signals using ML models."""
        try:
            if self.signal_service:
                return await self.signal_service.generate_trading_signals(
                    symbol=symbol, timeframe=timeframe, lookback_bars=
                    lookback_bars, models=models, context=context)
            from ml_integration_service.services.signal_generation import generate_signals
            return await generate_signals(symbol=symbol, timeframe=
                timeframe, lookback_bars=lookback_bars, models=models,
                context=context)
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
            if self.signal_service:
                return await self.signal_service.get_model_performance(model_id
                    =model_id, symbol=symbol, timeframe=timeframe,
                    start_date=start_date, end_date=end_date)
            from ml_integration_service.services.model_evaluation import get_performance_metrics
            return await get_performance_metrics(model_id=model_id, symbol=
                symbol, timeframe=timeframe, start_date=start_date,
                end_date=end_date)
        except Exception as e:
            logger.error(f'Error getting model performance: {str(e)}')
            return {'model_id': model_id, 'symbol': symbol, 'timeframe':
                timeframe, 'metrics': {}, 'error': str(e), 'is_fallback': True}
