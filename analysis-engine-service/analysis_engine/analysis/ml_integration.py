"""
Machine Learning Integration Layer

This module provides a comprehensive integration layer between the indicator system
and machine learning models. It handles:
- Feature preparation from indicators
- Model selection and loading
- Prediction generation and evaluation
- Feedback loops for continuous improvement
"""
import pandas as pd
import numpy as np
import joblib
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
import os
from datetime import datetime, timedelta
import time
import uuid
from pathlib import Path
import traceback
from concurrent.futures import ThreadPoolExecutor
from analysis_engine.analysis.indicator_interface import indicator_registry
from analysis_engine.analysis.feature_extraction import feature_store
from analysis_engine.analysis.signal_system import signal_system, SignalType
from common_lib.adapters.ml_integration_adapter import MLModelRegistryAdapter, MLJobTrackerAdapter, MLModelDeploymentAdapter, MLMetricsProviderAdapter
from common_lib.interfaces.ml_integration import PredictionHorizon, PredictionType, ModelStatus, ModelType, ModelFramework
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class ModelSource(Enum):
    """Sources of machine learning models"""
    LOCAL = 'local'
    MODEL_REGISTRY = 'registry'
    API = 'api'
    CUSTOM = 'custom'


class ModelFramework(Enum):
    """Supported ML frameworks"""
    SCIKIT_LEARN = 'sklearn'
    TENSORFLOW = 'tensorflow'
    PYTORCH = 'pytorch'
    XGB = 'xgboost'
    LIGHTGBM = 'lightgbm'
    CUSTOM = 'custom'


@dataclass
class ModelConfiguration:
    """Configuration for a machine learning model"""
    id: str
    name: str
    description: str
    source: ModelSource
    framework: ModelFramework
    model_path: str
    version: str = '1.0.0'
    features: List[str] = field(default_factory=list)
    prediction_type: Optional[PredictionType] = None
    horizon: Optional[PredictionHorizon] = None
    instruments: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    preprocessing_steps: List[Dict[str, Any]] = field(default_factory=list)
    api_config: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self):
        """Initialize with defaults if needed"""
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class ModelPrediction:
    """Prediction result from a model"""
    id: str
    model_id: str
    model_name: str
    timestamp: datetime
    instrument: str
    timeframe: str
    prediction_type: PredictionType
    horizon: PredictionHorizon
    values: Dict[str, Any]
    confidence: float
    features_used: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize with defaults if needed"""
        if not self.id:
            self.id = str(uuid.uuid4())

    @property
    def expiration(self) ->datetime:
        """Get the expiration time of this prediction"""
        horizon_seconds = self.horizon.value
        return self.timestamp + timedelta(seconds=horizon_seconds)

    @property
    def is_expired(self) ->bool:
        """Check if the prediction has expired"""
        return datetime.now() > self.expiration

    def to_dict(self) ->Dict[str, Any]:
        """Convert to dictionary"""
        return {'id': self.id, 'model_id': self.model_id, 'model_name':
            self.model_name, 'timestamp': self.timestamp.isoformat(),
            'instrument': self.instrument, 'timeframe': self.timeframe,
            'prediction_type': self.prediction_type.name, 'horizon': self.
            horizon.name, 'values': self.values, 'confidence': self.
            confidence, 'features_used': self.features_used, 'expires_at':
            self.expiration.isoformat(), 'metadata': self.metadata}


class ModelLoadError(Exception):
    """Exception raised when a model cannot be loaded"""
    pass


class FeaturePreparationError(Exception):
    """Exception raised when features cannot be prepared"""
    pass


class PredictionError(Exception):
    """Exception raised when prediction generation fails"""
    pass


class IndicatorMLBridge:
    """
    Bridge between indicators and machine learning models.

    This class handles the preparation of indicator data for use with ML models,
    loading and managing models, and generating predictions.
    """

    def __init__(self, model_configs: List[ModelConfiguration]=None):
        """
        Initialize the indicator-ML bridge

        Args:
            model_configs: List of model configurations to load
        """
        self._models: Dict[str, Any] = {}
        self._model_configs: Dict[str, ModelConfiguration] = {}
        self._predictions: List[ModelPrediction] = []
        if model_configs:
            for config in model_configs:
                self.add_model_config(config)

    def add_model_config(self, config: ModelConfiguration) ->None:
        """
        Add a model configuration

        Args:
            config: The model configuration to add
        """
        self._model_configs[config.id] = config
        logger.info(f'Added model configuration: {config.name} ({config.id})')

    @with_database_resilience('load_model_configs')
    @with_exception_handling
    def load_model_configs(self, config_path: str) ->int:
        """
        Load model configurations from a JSON file

        Args:
            config_path: Path to the JSON configuration file

        Returns:
            Number of configurations loaded
        """
        try:
            with open(config_path, 'r') as f:
                configs_data = json.load(f)
            count = 0
            for config_data in configs_data:
                try:
                    config_data['source'] = ModelSource(config_data.get(
                        'source', 'local'))
                    config_data['framework'] = ModelFramework(config_data.
                        get('framework', 'sklearn'))
                    if 'prediction_type' in config_data:
                        config_data['prediction_type'] = PredictionType[
                            config_data['prediction_type']]
                    if 'horizon' in config_data:
                        config_data['horizon'] = PredictionHorizon[config_data
                            ['horizon']]
                    config = ModelConfiguration(**config_data)
                    self.add_model_config(config)
                    count += 1
                except Exception as e:
                    logger.error(f'Error loading model configuration: {str(e)}'
                        )
                    continue
            logger.info(
                f'Loaded {count} model configurations from {config_path}')
            return count
        except Exception as e:
            logger.error(
                f'Error loading model configurations from {config_path}: {str(e)}'
                )
            return 0

    @with_exception_handling
    def save_model_configs(self, config_path: str) ->bool:
        """
        Save model configurations to a JSON file

        Args:
            config_path: Path to save the JSON configuration

        Returns:
            True if successful, False otherwise
        """
        try:
            configs_data = []
            for config in self._model_configs.values():
                config_dict = {'id': config.id, 'name': config.name,
                    'description': config.description, 'source': config.
                    source.value, 'framework': config.framework.value,
                    'model_path': config.model_path, 'version': config.
                    version, 'features': config.features, 'prediction_type':
                    config.prediction_type.name if config.prediction_type else
                    None, 'horizon': config.horizon.name if config.horizon else
                    None, 'instruments': config.instruments, 'timeframes':
                    config.timeframes, 'preprocessing_steps': config.
                    preprocessing_steps, 'api_config': config.api_config,
                    'parameters': config.parameters, 'metrics': config.
                    metrics, 'enabled': config.enabled}
                configs_data.append(config_dict)
            with open(config_path, 'w') as f:
                json.dump(configs_data, f, indent=4)
            logger.info(
                f'Saved {len(configs_data)} model configurations to {config_path}'
                )
            return True
        except Exception as e:
            logger.error(
                f'Error saving model configurations to {config_path}: {str(e)}'
                )
            return False

    @with_resilience('get_model_config')
    def get_model_config(self, model_id: str) ->Optional[ModelConfiguration]:
        """
        Get a model configuration by ID

        Args:
            model_id: The ID of the model configuration

        Returns:
            The model configuration, or None if not found
        """
        return self._model_configs.get(model_id)

    @with_database_resilience('load_model')
    @with_exception_handling
    def load_model(self, model_id: str) ->Any:
        """
        Load a model by its configuration ID

        Args:
            model_id: The ID of the model to load

        Returns:
            The loaded model object

        Raises:
            ModelLoadError: If the model cannot be loaded
        """
        if model_id in self._models:
            return self._models[model_id]
        config = self.get_model_config(model_id)
        if not config:
            raise ModelLoadError(f'Model configuration not found: {model_id}')
        if not config.enabled:
            raise ModelLoadError(
                f'Model is disabled: {config.name} ({model_id})')
        try:
            model = None
            if config.source == ModelSource.LOCAL:
                if config.framework == ModelFramework.SCIKIT_LEARN:
                    model = joblib.load(config.model_path)
                elif config.framework == ModelFramework.TENSORFLOW:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(config.model_path)
                elif config.framework == ModelFramework.PYTORCH:
                    import torch
                    model = torch.load(config.model_path)
                elif config.framework == ModelFramework.XGB:
                    import xgboost as xgb
                    model = xgb.Booster()
                    model.load_model(config.model_path)
                elif config.framework == ModelFramework.LIGHTGBM:
                    import lightgbm as lgb
                    model = lgb.Booster(model_file=config.model_path)
                elif config.framework == ModelFramework.CUSTOM:
                    raise NotImplementedError(
                        'Custom model loading not implemented yet')
                else:
                    raise ModelLoadError(
                        f'Unsupported framework: {config.framework}')
            elif config.source == ModelSource.MODEL_REGISTRY:
                try:
                    model_registry_adapter = MLModelRegistryAdapter()
                    model_details = await model_registry_adapter.get_model(
                        config.model_path)
                    if model_details.get('status') != ModelStatus.ACTIVE.value:
                        raise ModelLoadError(
                            f'Model {config.model_path} is not active')
                    model_path = model_details.get('storage_path')
                    if not model_path:
                        raise ModelLoadError(
                            f'Model {config.model_path} has no storage path')
                    if config.framework == ModelFramework.SCIKIT_LEARN:
                        model = joblib.load(model_path)
                    elif config.framework == ModelFramework.TENSORFLOW:
                        import tensorflow as tf
                        model = tf.keras.models.load_model(model_path)
                    elif config.framework == ModelFramework.PYTORCH:
                        import torch
                        model = torch.load(model_path)
                    elif config.framework == ModelFramework.XGB:
                        import xgboost as xgb
                        model = xgb.Booster()
                        model.load_model(model_path)
                    elif config.framework == ModelFramework.LIGHTGBM:
                        import lightgbm as lgb
                        model = lgb.Booster(model_file=model_path)
                    else:
                        raise ModelLoadError(
                            f'Unsupported framework for model registry: {config.framework}'
                            )
                    model_metadata = model_details.get('metadata', {})
                    config.parameters.update(model_metadata)
                    logger.info(f'Loaded model {config.name} from registry')
                except Exception as e:
                    raise ModelLoadError(
                        f'Failed to load model from registry: {str(e)}') from e
            elif config.source == ModelSource.API:
                model = {'type': 'api_model', 'config': config.api_config}
            elif config.source == ModelSource.CUSTOM:
                raise NotImplementedError(
                    'Custom model source loading not implemented yet')
            else:
                raise ModelLoadError(
                    f'Unsupported model source: {config.source}')
            self._models[model_id] = model
            logger.info(f'Loaded model: {config.name} ({model_id})')
            return model
        except Exception as e:
            error_message = f'Failed to load model {model_id}: {str(e)}'
            logger.error(error_message)
            logger.error(traceback.format_exc())
            raise ModelLoadError(error_message) from e

    @with_exception_handling
    def prepare_features(self, data: pd.DataFrame, model_id: str
        ) ->pd.DataFrame:
        """
        Prepare features for a specific model

        Args:
            data: Input data (typically price/volume data)
            model_id: ID of the model to prepare features for

        Returns:
            DataFrame with prepared features

        Raises:
            FeaturePreparationError: If features cannot be prepared
        """
        try:
            config = self.get_model_config(model_id)
            if not config:
                raise FeaturePreparationError(
                    f'Model configuration not found: {model_id}')
            required_features = config.features
            if not required_features:
                raise FeaturePreparationError(
                    f'No features specified for model: {model_id}')
            indicator_data = data.copy()
            indicators_needed = set()
            for feature in required_features:
                if '.' in feature:
                    indicator_name = feature.split('.')[0]
                    indicators_needed.add(indicator_name)
            for indicator_name in indicators_needed:
                try:
                    result = indicator_registry.calculate_indicator(
                        indicator_name, data, use_cache=True)
                    for col in result.data.columns:
                        if col not in indicator_data.columns:
                            indicator_data[col] = result.data[col]
                except Exception as e:
                    logger.warning(
                        f'Failed to calculate indicator {indicator_name}: {str(e)}'
                        )
            if hasattr(feature_store, 'extract_features'):
                feature_names = [f.split('.')[-1] for f in required_features]
                feature_data = feature_store.extract_features(indicator_data,
                    feature_names)
            else:
                feature_data = pd.DataFrame(index=indicator_data.index)
                for feature in required_features:
                    if '.' in feature:
                        indicator_name, column_name = feature.split('.', 1)
                        if column_name in indicator_data.columns:
                            feature_data[feature] = indicator_data[column_name]
                    elif feature in indicator_data.columns:
                        feature_data[feature] = indicator_data[feature]
            for step in config.preprocessing_steps:
                step_type = step.get('type')
                if step_type == 'fill_na':
                    method = step.get('method', 'ffill')
                    feature_data = feature_data.fillna(method=method)
                elif step_type == 'scale':
                    method = step.get('method', 'min_max')
                    if method == 'min_max':
                        for col in feature_data.columns:
                            min_val = feature_data[col].min()
                            max_val = feature_data[col].max()
                            if max_val > min_val:
                                feature_data[col] = (feature_data[col] -
                                    min_val) / (max_val - min_val)
                elif step_type == 'lag':
                    periods = step.get('periods', 1)
                    feature_data = feature_data.shift(periods)
            if feature_data.isna().any().any():
                feature_data = feature_data.fillna(0)
                logger.warning(
                    f'Filled NaN values in features for model {model_id}')
            return feature_data
        except Exception as e:
            error_message = (
                f'Failed to prepare features for model {model_id}: {str(e)}')
            logger.error(error_message)
            logger.error(traceback.format_exc())
            raise FeaturePreparationError(error_message) from e

    @with_exception_handling
    def predict(self, data: pd.DataFrame, model_id: str, instrument: str,
        timeframe: str) ->ModelPrediction:
        """
        Generate a prediction using a specific model

        Args:
            data: Input data (price/volume data)
            model_id: ID of the model to use
            instrument: The instrument the prediction is for
            timeframe: The timeframe the prediction is for

        Returns:
            ModelPrediction object with the prediction

        Raises:
            PredictionError: If the prediction cannot be generated
        """
        try:
            config = self.get_model_config(model_id)
            if not config:
                raise PredictionError(
                    f'Model configuration not found: {model_id}')
            if config.instruments and instrument not in config.instruments:
                raise PredictionError(
                    f'Model {model_id} not configured for instrument {instrument}'
                    )
            if config.timeframes and timeframe not in config.timeframes:
                raise PredictionError(
                    f'Model {model_id} not configured for timeframe {timeframe}'
                    )
            features = self.prepare_features(data, model_id)
            model = self.load_model(model_id)
            prediction_values = {}
            confidence = 0.5
            latest_features = features.iloc[-1].to_dict()
            features_used = latest_features.copy()
            if config.source == ModelSource.API:
                api_config = config.api_config
                url = api_config_manager.get('url')
                if not url:
                    raise PredictionError('API URL not configured')
                import re
                from urllib.parse import urlparse
                parsed_url = urlparse(url)
                if parsed_url.scheme not in ['http', 'https']:
                    raise PredictionError(
                        f'Unsupported URL protocol: {parsed_url.scheme}')
                hostname = parsed_url.netloc.split(':')[0]
                if hostname in ['localhost', '127.0.0.1', '::1'
                    ] or hostname.startswith('192.168.'
                    ) or hostname.startswith('10.') or hostname.startswith(
                    '172.') and 16 <= int(hostname.split('.')[1]) <= 31:
                    raise PredictionError(
                        f'Access to internal network addresses is not allowed: {hostname}'
                        )
                import requests
                request_data = {'features': latest_features, 'instrument':
                    instrument, 'timeframe': timeframe}
                headers = api_config_manager.get('headers', {})
                response = requests.post(url, json=request_data, headers=
                    headers, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    prediction_values = result.get('prediction', {})
                    confidence = result.get('confidence', 0.5)
                else:
                    raise PredictionError(
                        f'API request failed: {response.status_code}')
            else:
                feature_array = features.iloc[-1:].values
                if config.framework == ModelFramework.SCIKIT_LEARN:
                    if hasattr(model, 'predict_proba'):
                        probas = model.predict_proba(feature_array)[0]
                        pred_class = model.classes_[np.argmax(probas)]
                        confidence = np.max(probas)
                        prediction_values = {'class': str(pred_class),
                            'probability': float(confidence)}
                        if len(model.classes_) == 2:
                            prediction_values['direction'
                                ] = 1 if pred_class == 1 else -1
                    else:
                        pred_value = model.predict(feature_array)[0]
                        if (config.prediction_type == PredictionType.
                            PRICE_DIRECTION):
                            prediction_values = {'direction': 1 if 
                                pred_value > 0 else -1 if pred_value < 0 else
                                0, 'value': float(pred_value)}
                        else:
                            prediction_values = {'value': float(pred_value)}
                elif config.framework == ModelFramework.TENSORFLOW:
                    pred_array = model.predict(feature_array)
                    if len(pred_array.shape) > 1 and pred_array.shape[1] > 1:
                        pred_class = np.argmax(pred_array[0])
                        confidence = pred_array[0][pred_class]
                        prediction_values = {'class': int(pred_class),
                            'probability': float(confidence)}
                    else:
                        pred_value = float(pred_array[0][0])
                        if (config.prediction_type == PredictionType.
                            PRICE_DIRECTION):
                            threshold = config.parameters.get('threshold', 0.5)
                            direction = (1 if pred_value > threshold else -
                                1 if pred_value < 1 - threshold else 0)
                            prediction_values = {'direction': direction,
                                'probability': pred_value}
                        else:
                            prediction_values = {'value': pred_value}
                elif config.framework == ModelFramework.PYTORCH:
                    import torch
                    X_tensor = torch.tensor(feature_array, dtype=torch.float32)
                    model.eval()
                    with torch.no_grad():
                        pred_tensor = model(X_tensor)
                    pred_array = pred_tensor.numpy()
                    if (config.prediction_type == PredictionType.
                        PRICE_DIRECTION):
                        if pred_array.shape[1] > 1:
                            pred_class = np.argmax(pred_array[0])
                            confidence = pred_array[0][pred_class]
                            direction_map = {(0): -1, (1): 0, (2): 1}
                            direction = direction_map.get(pred_class, 0)
                            prediction_values = {'direction': direction,
                                'class': int(pred_class), 'probability':
                                float(confidence)}
                        else:
                            pred_value = float(pred_array[0][0])
                            threshold = config.parameters.get('threshold', 0.5)
                            direction = 1 if pred_value > threshold else -1
                            prediction_values = {'direction': direction,
                                'probability': pred_value}
                    else:
                        prediction_values = {'value': float(pred_array[0][0])}
                elif config.framework in (ModelFramework.XGB,
                    ModelFramework.LIGHTGBM):
                    if config.framework == ModelFramework.XGB:
                        import xgboost as xgb
                        dmatrix = xgb.DMatrix(feature_array)
                        pred_array = model.predict(dmatrix)
                    else:
                        pred_array = model.predict(feature_array)
                    if len(pred_array.shape) > 1 and pred_array.shape[1] > 1:
                        pred_class = np.argmax(pred_array[0])
                        confidence = pred_array[0][pred_class]
                        prediction_values = {'class': int(pred_class),
                            'probability': float(confidence)}
                    else:
                        pred_value = float(pred_array[0])
                        if (config.prediction_type == PredictionType.
                            PRICE_DIRECTION):
                            threshold = config.parameters.get('threshold', 0.5)
                            direction = (1 if pred_value > threshold else -
                                1 if pred_value < 1 - threshold else 0)
                            prediction_values = {'direction': direction,
                                'probability': pred_value}
                        else:
                            prediction_values = {'value': pred_value}
            prediction = ModelPrediction(id=str(uuid.uuid4()), model_id=
                model_id, model_name=config.name, timestamp=datetime.now(),
                instrument=instrument, timeframe=timeframe, prediction_type
                =config.prediction_type or PredictionType.PRICE_DIRECTION,
                horizon=config.horizon or PredictionHorizon.HOUR_1, values=
                prediction_values, confidence=confidence, features_used=
                features_used, metadata={'feature_count': len(features.
                columns)})
            self._predictions.append(prediction)
            if len(self._predictions) > 1000:
                self._predictions = self._predictions[-1000:]
            return prediction
        except Exception as e:
            error_message = f'Prediction failed for model {model_id}: {str(e)}'
            logger.error(error_message)
            logger.error(traceback.format_exc())
            raise PredictionError(error_message) from e

    @with_exception_handling
    def predict_multiple(self, data: pd.DataFrame, model_ids: List[str],
        instrument: str, timeframe: str, parallel: bool=True) ->List[
        ModelPrediction]:
        """
        Generate predictions using multiple models

        Args:
            data: Input data (price/volume data)
            model_ids: List of model IDs to use
            instrument: The instrument the predictions are for
            timeframe: The timeframe the predictions are for
            parallel: Whether to predict in parallel

        Returns:
            List of ModelPrediction objects
        """
        predictions = []
        if parallel and len(model_ids) > 1:
            with ThreadPoolExecutor() as executor:
                futures = []
                for model_id in model_ids:
                    future = executor.submit(self.predict, data, model_id,
                        instrument, timeframe)
                    futures.append((model_id, future))
                for model_id, future in futures:
                    try:
                        prediction = future.result()
                        predictions.append(prediction)
                    except Exception as e:
                        logger.error(
                            f'Prediction failed for model {model_id}: {str(e)}'
                            )
        else:
            for model_id in model_ids:
                try:
                    prediction = self.predict(data, model_id, instrument,
                        timeframe)
                    predictions.append(prediction)
                except Exception as e:
                    logger.error(
                        f'Prediction failed for model {model_id}: {str(e)}')
        return predictions

    @with_analysis_resilience('get_recent_predictions')
    def get_recent_predictions(self, instrument: Optional[str]=None,
        timeframe: Optional[str]=None, model_id: Optional[str]=None,
        prediction_type: Optional[PredictionType]=None, max_age: Optional[
        timedelta]=None) ->List[ModelPrediction]:
        """
        Get recent predictions, optionally filtered

        Args:
            instrument: Filter by instrument
            timeframe: Filter by timeframe
            model_id: Filter by model ID
            prediction_type: Filter by prediction type
            max_age: Maximum age of predictions to include

        Returns:
            List of matching prediction objects
        """
        predictions = self._predictions.copy()
        if instrument:
            predictions = [p for p in predictions if p.instrument == instrument
                ]
        if timeframe:
            predictions = [p for p in predictions if p.timeframe == timeframe]
        if model_id:
            predictions = [p for p in predictions if p.model_id == model_id]
        if prediction_type:
            predictions = [p for p in predictions if p.prediction_type ==
                prediction_type]
        if max_age:
            min_timestamp = datetime.now() - max_age
            predictions = [p for p in predictions if p.timestamp >=
                min_timestamp]
        return sorted(predictions, key=lambda p: p.timestamp, reverse=True)

    def generate_signals_from_predictions(self, predictions: List[
        ModelPrediction], confidence_threshold: float=0.6,
        use_signal_system: bool=True) ->List[Dict[str, Any]]:
        """
        Generate trading signals from model predictions

        Args:
            predictions: List of model predictions
            confidence_threshold: Minimum confidence for a signal
            use_signal_system: Whether to use the signal system

        Returns:
            List of generated signals
        """
        signals = []
        for pred in predictions:
            if pred.is_expired or pred.confidence < confidence_threshold:
                continue
            if pred.prediction_type == PredictionType.PRICE_DIRECTION:
                direction = pred.values.get('direction')
                if direction == 1:
                    signal_type = SignalType.BUY
                elif direction == -1:
                    signal_type = SignalType.SELL
                else:
                    continue
                if use_signal_system:
                    from analysis_engine.analysis.signal_system import Signal
                    signal = Signal(timestamp=pred.timestamp,
                        indicator_name=f'ML:{pred.model_name}', signal_type
                        =signal_type, strength=pred.confidence, price=pred.
                        features_used.get('close', 0.0), metadata={
                        'model_id': pred.model_id, 'prediction_horizon':
                        pred.horizon.label, 'prediction_id': pred.id})
                    signal_system.add_signal(signal)
                    signals.append({'id': str(uuid.uuid4()),
                        'prediction_id': pred.id, 'model_id': pred.model_id,
                        'instrument': pred.instrument, 'timeframe': pred.
                        timeframe, 'signal_type': signal_type.name,
                        'strength': pred.confidence, 'timestamp': pred.
                        timestamp.isoformat()})
                else:
                    signals.append({'id': str(uuid.uuid4()),
                        'prediction_id': pred.id, 'model_id': pred.model_id,
                        'instrument': pred.instrument, 'timeframe': pred.
                        timeframe, 'signal_type': 'BUY' if direction == 1 else
                        'SELL', 'strength': pred.confidence, 'timestamp':
                        pred.timestamp.isoformat()})
        return signals

    @with_exception_handling
    def save_predictions_to_file(self, filepath: str) ->bool:
        """
        Save recent predictions to a JSON file

        Args:
            filepath: Path to save the predictions

        Returns:
            True if successful, False otherwise
        """
        try:
            import os
            from pathlib import Path
            normalized_path = os.path.normpath(os.path.abspath(filepath))
            allowed_dirs = [os.path.abspath('./predictions'), os.path.
                abspath('./data/predictions'), os.path.abspath('./output')]
            is_allowed = False
            for allowed_dir in allowed_dirs:
                if normalized_path.startswith(allowed_dir):
                    is_allowed = True
                    break
            if not is_allowed:
                logger.error(
                    f'Path not allowed for saving predictions: {filepath}')
                return False
            os.makedirs(os.path.dirname(normalized_path), exist_ok=True)
            predictions_data = [p.to_dict() for p in self._predictions]
            with open(normalized_path, 'w') as f:
                json.dump(predictions_data, f, indent=4)
            logger.info(
                f'Saved {len(predictions_data)} predictions to {normalized_path}'
                )
            return True
        except Exception as e:
            logger.error(f'Failed to save predictions to {filepath}: {str(e)}')
            return False


ml_bridge = IndicatorMLBridge()
