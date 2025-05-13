"""
Model Serving Engine Module.

This module provides functionality for serving ML models with low-latency
and high-reliability requirements. It includes versioning, canary deployment,
and monitoring capabilities.
"""
import os
import time
import logging
import threading
from typing import Dict, Any, List, Optional, Union, Callable
import numpy as np
from datetime import datetime
from ml_workbench_service.model_registry.model_registry import ModelRegistry
from ml_workbench_service.services.model_monitor import ModelMonitor
from ml_workbench_service.models.model import ModelVersion, ModelMetadata, PredictionRequest
logger = logging.getLogger(__name__)


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ModelServingEngine:
    """
    Engine for serving ML models with low-latency inference.
    
    This class provides functionality to:
    - Load models from the model registry
    - Serve predictions with low latency
    - Support model versioning and canary deployments
    - Monitor inference performance and errors
    - Handle model reloading and updates
    """

    def __init__(self, model_registry: ModelRegistry, model_monitor:
        Optional[ModelMonitor]=None, cache_size: int=10,
        model_refresh_interval_seconds: int=300):
        """
        Initialize the model serving engine.
        
        Args:
            model_registry: Registry to load models from
            model_monitor: Monitor for tracking model performance (optional)
            cache_size: Number of models to keep loaded in memory
            model_refresh_interval_seconds: Interval for checking model updates
        """
        self.model_registry = model_registry
        self.model_monitor = model_monitor
        self.cache_size = cache_size
        self.model_refresh_interval_seconds = model_refresh_interval_seconds
        self.model_cache: Dict[str, Any] = {}
        self.model_metadata_cache: Dict[str, ModelMetadata] = {}
        self.model_last_used: Dict[str, datetime] = {}
        self.traffic_routing: Dict[str, Dict[str, Any]] = {}
        self._stop_refresh_thread = False
        self._refresh_thread = threading.Thread(target=self.
            _refresh_models_periodically)
        self._refresh_thread.daemon = True
        self._refresh_thread.start()
        logger.info('ModelServingEngine initialized')

    def load_model(self, model_name: str, version_id: Optional[str]=None
        ) ->str:
        """
        Load a model from the registry into memory.
        
        Args:
            model_name: Name of the model
            version_id: Specific version ID to load (or latest production if None)
            
        Returns:
            Version ID of the loaded model
            
        Raises:
            ValueError: If model or version doesn't exist
        """
        if version_id is None:
            model_version = self.model_registry.get_latest_version(model_name,
                stage='production')
            if not model_version:
                raise ValueError(
                    f'No production version found for model {model_name}')
            version_id = model_version.version_id
        else:
            model_version = self.model_registry.get_model_version(model_name,
                version_id)
            if not model_version:
                raise ValueError(
                    f'Model version {version_id} not found for {model_name}')
        cache_key = f'{model_name}:{version_id}'
        if cache_key in self.model_cache:
            self.model_last_used[cache_key] = datetime.now()
            logger.debug(f'Model {cache_key} already loaded and cached')
            return version_id
        if len(self.model_cache) >= self.cache_size:
            self._evict_least_used_model()
        logger.info(f'Loading model {model_name}, version {version_id}')
        start_time = time.time()
        model = self.model_registry.load_model(model_name, version_id)
        metadata = self.model_registry.get_model_metadata(model_name,
            version_id)
        load_time = time.time() - start_time
        logger.info(
            f'Model {model_name}, version {version_id} loaded in {load_time:.2f} seconds'
            )
        self.model_cache[cache_key] = model
        self.model_metadata_cache[cache_key] = metadata
        self.model_last_used[cache_key] = datetime.now()
        return version_id

    def setup_canary_deployment(self, model_name: str,
        production_version_id: str, canary_version_id: str, canary_percent:
        int=10) ->None:
        """
        Setup a canary deployment for a model.
        
        Args:
            model_name: Model name
            production_version_id: Version ID for main production traffic
            canary_version_id: Version ID for canary traffic
            canary_percent: Percentage of traffic to route to canary (0-100)
            
        Raises:
            ValueError: If model versions don't exist
        """
        self.load_model(model_name, production_version_id)
        self.load_model(model_name, canary_version_id)
        self.traffic_routing[model_name] = {'production':
            production_version_id, 'canary': canary_version_id,
            'canary_percent': max(0, min(100, canary_percent))}
        logger.info(
            f'Canary deployment set: {model_name} production={production_version_id}, canary={canary_version_id}, split={canary_percent}%'
            )

    @with_exception_handling
    def predict(self, model_name: str, inputs: Dict[str, Any], version_id:
        Optional[str]=None) ->Dict[str, Any]:
        """
        Get a prediction from a model.
        
        Args:
            model_name: Name of the model
            inputs: Input data for prediction 
            version_id: Specific version to use (if None, uses routing rules)
            
        Returns:
            Prediction results as a dictionary
            
        Raises:
            ValueError: If model can't be found or loaded
        """
        start_time = time.time()
        prediction_id = f'{model_name}_{int(start_time * 1000)}'
        if version_id is None:
            version_id = self._get_routed_version_id(model_name)
        cache_key = f'{model_name}:{version_id}'
        if cache_key not in self.model_cache:
            self.load_model(model_name, version_id)
        self.model_last_used[cache_key] = datetime.now()
        model = self.model_cache[cache_key]
        metadata = self.model_metadata_cache[cache_key]
        prediction_request = PredictionRequest(id=prediction_id, model_name
            =model_name, version_id=version_id, timestamp=datetime.now(),
            inputs=inputs)
        try:
            prediction = self._run_prediction(model, inputs, metadata)
            duration_ms = (time.time() - start_time) * 1000
            if self.model_monitor:
                self.model_monitor.record_prediction(prediction_request=
                    prediction_request, outputs=prediction, duration_ms=
                    duration_ms)
            result = {'prediction': prediction, 'metadata': {'model_name':
                model_name, 'version_id': version_id, 'prediction_id':
                prediction_id, 'duration_ms': duration_ms}}
            return result
        except Exception as e:
            if self.model_monitor:
                self.model_monitor.record_prediction_error(prediction_request
                    =prediction_request, error=str(e))
            logger.error(
                f'Error making prediction with {model_name}, version {version_id}: {str(e)}'
                )
            raise

    @with_exception_handling
    def batch_predict(self, model_name: str, batch_inputs: List[Dict[str,
        Any]], version_id: Optional[str]=None) ->List[Dict[str, Any]]:
        """
        Get batch predictions from a model.
        
        Args:
            model_name: Name of the model
            batch_inputs: List of input data for predictions
            version_id: Specific version to use (if None, uses routing rules)
            
        Returns:
            List of prediction results
        """
        results = []
        for inputs in batch_inputs:
            try:
                result = self.predict(model_name, inputs, version_id)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e), 'inputs': inputs,
                    'metadata': {'model_name': model_name, 'version_id': 
                    version_id if version_id else 'unknown'}})
        return results

    def get_model_info(self, model_name: str, version_id: Optional[str]=None
        ) ->Dict[str, Any]:
        """
        Get information about a loaded model.
        
        Args:
            model_name: Name of the model
            version_id: Specific version ID (or latest production if None)
            
        Returns:
            Dictionary with model information
            
        Raises:
            ValueError: If model doesn't exist
        """
        if version_id is None:
            model_version = self.model_registry.get_latest_version(model_name,
                stage='production')
            if not model_version:
                raise ValueError(
                    f'No production version found for model {model_name}')
            version_id = model_version.version_id
        routing_info = {}
        if model_name in self.traffic_routing:
            routing = self.traffic_routing[model_name]
            routing_info = {'production_version': routing['production'],
                'canary_version': routing.get('canary'), 'canary_percent':
                routing.get('canary_percent', 0)}
        model_metadata = self.model_registry.get_model_metadata(model_name,
            version_id)
        cache_key = f'{model_name}:{version_id}'
        is_loaded = cache_key in self.model_cache
        performance_metrics = {}
        if self.model_monitor:
            performance_metrics = self.model_monitor.get_model_metrics(
                model_name, version_id)
        return {'model_name': model_name, 'version_id': version_id,
            'is_loaded': is_loaded, 'metadata': model_metadata.to_dict() if
            model_metadata else {}, 'routing_info': routing_info,
            'performance_metrics': performance_metrics}

    def list_loaded_models(self) ->List[Dict[str, Any]]:
        """
        List all models currently loaded in the serving engine.
        
        Returns:
            List of dictionaries with loaded model info
        """
        result = []
        for cache_key in self.model_cache:
            model_name, version_id = cache_key.split(':')
            last_used = self.model_last_used.get(cache_key)
            result.append({'model_name': model_name, 'version_id':
                version_id, 'last_used': last_used.isoformat() if last_used
                 else None})
        return result

    def unload_model(self, model_name: str, version_id: str) ->None:
        """
        Unload a model from memory.
        
        Args:
            model_name: Name of the model
            version_id: Version ID to unload
            
        Raises:
            ValueError: If model not found in cache
        """
        cache_key = f'{model_name}:{version_id}'
        if cache_key not in self.model_cache:
            raise ValueError(
                f'Model {model_name}, version {version_id} is not loaded')
        del self.model_cache[cache_key]
        if cache_key in self.model_metadata_cache:
            del self.model_metadata_cache[cache_key]
        if cache_key in self.model_last_used:
            del self.model_last_used[cache_key]
        if model_name in self.traffic_routing:
            routing = self.traffic_routing[model_name]
            if routing.get('production') == version_id or routing.get('canary'
                ) == version_id:
                logger.warning(
                    f'Unloading model {model_name}:{version_id} that is part of traffic routing'
                    )
                del self.traffic_routing[model_name]
        logger.info(f'Unloaded model {model_name}, version {version_id}')

    def shutdown(self) ->None:
        """Gracefully shut down the serving engine."""
        logger.info('Shutting down ModelServingEngine')
        self._stop_refresh_thread = True
        if hasattr(self, '_refresh_thread') and self._refresh_thread.is_alive(
            ):
            self._refresh_thread.join(timeout=10)
        self.model_cache = {}
        self.model_metadata_cache = {}
        self.model_last_used = {}

    @with_exception_handling
    def _run_prediction(self, model: Any, inputs: Dict[str, Any], metadata:
        ModelMetadata) ->Dict[str, Any]:
        """Run prediction on a model with the provided inputs."""
        try:
            processed_inputs = self._preprocess_inputs(inputs, metadata)
            raw_prediction = model.predict(processed_inputs)
            prediction = self._postprocess_outputs(raw_prediction, metadata)
            return prediction
        except Exception as e:
            logger.error(f'Prediction error: {str(e)}')
            raise

    def _preprocess_inputs(self, inputs: Dict[str, Any], metadata:
        ModelMetadata) ->Dict[str, Any]:
        """Preprocess inputs based on model metadata."""
        return inputs

    def _postprocess_outputs(self, raw_prediction: Any, metadata: ModelMetadata
        ) ->Dict[str, Any]:
        """Postprocess raw model outputs based on model metadata."""
        if isinstance(raw_prediction, (np.ndarray, list, tuple)):
            if len(raw_prediction) == 1:
                return {'prediction': raw_prediction[0]}
            else:
                return {'predictions': raw_prediction}
        elif isinstance(raw_prediction, dict):
            return raw_prediction
        else:
            return {'prediction': raw_prediction}

    def _get_routed_version_id(self, model_name: str) ->str:
        """
        Get the appropriate version ID based on routing rules.
        
        Args:
            model_name: Model name
            
        Returns:
            Version ID to use
            
        Raises:
            ValueError: If model doesn't have routing setup
        """
        if model_name not in self.traffic_routing:
            model_version = self.model_registry.get_latest_version(model_name,
                stage='production')
            if not model_version:
                raise ValueError(
                    f'No production version found for model {model_name}')
            return model_version.version_id
        routing = self.traffic_routing[model_name]
        canary_percent = routing.get('canary_percent', 0)
        if canary_percent > 0 and np.random.random() * 100 < canary_percent:
            return routing['canary']
        else:
            return routing['production']

    def _evict_least_used_model(self) ->None:
        """
        Evict the least recently used model from cache.
        """
        if not self.model_last_used:
            return
        lru_key = min(self.model_last_used.items(), key=lambda x: x[1])[0]
        model_name, version_id = lru_key.split(':')
        is_routed = False
        if model_name in self.traffic_routing:
            routing = self.traffic_routing[model_name]
            if routing.get('production') == version_id or routing.get('canary'
                ) == version_id:
                is_routed = True
        if is_routed:
            for key in self.model_last_used:
                m_name, v_id = key.split(':')
                is_key_routed = False
                if m_name in self.traffic_routing:
                    r = self.traffic_routing[m_name]
                    if r.get('production') == v_id or r.get('canary') == v_id:
                        is_key_routed = True
                if not is_key_routed:
                    lru_key = key
                    break
            if is_routed:
                lru_key = min(self.model_last_used.items(), key=lambda x: x[1]
                    )[0]
        logger.info(f'Evicting model from cache: {lru_key}')
        if lru_key in self.model_cache:
            del self.model_cache[lru_key]
        if lru_key in self.model_metadata_cache:
            del self.model_metadata_cache[lru_key]
        if lru_key in self.model_last_used:
            del self.model_last_used[lru_key]

    @with_exception_handling
    def _refresh_models_periodically(self) ->None:
        """
        Background thread to periodically check for model updates.
        """
        while not self._stop_refresh_thread:
            try:
                for model_name, routing in list(self.traffic_routing.items()):
                    production_version_id = routing.get('production')
                    latest_version = self.model_registry.get_latest_version(
                        model_name, stage='production')
                    if (latest_version and latest_version.version_id !=
                        production_version_id):
                        logger.info(
                            f'New production version available for {model_name}: {latest_version.version_id} (current: {production_version_id})'
                            )
                        try:
                            self.load_model(model_name, latest_version.
                                version_id)
                        except Exception as e:
                            logger.error(
                                f'Failed to pre-load new version: {str(e)}')
            except Exception as e:
                logger.error(f'Error in model refresh thread: {str(e)}')
            for _ in range(self.model_refresh_interval_seconds):
                if self._stop_refresh_thread:
                    break
                time.sleep(1)
