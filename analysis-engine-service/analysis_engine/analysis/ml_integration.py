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

# Local imports
from analysis_engine.analysis.indicator_interface import indicator_registry
from analysis_engine.analysis.feature_extraction import feature_store
from analysis_engine.analysis.signal_system import signal_system, SignalType
from ml_integration_service.model_connector import (
    PredictionHorizon, 
    PredictionType, 
    ModelMetadata
)

# Configure logging
logger = logging.getLogger(__name__)


class ModelSource(Enum):
    """Sources of machine learning models"""
    LOCAL = "local"           # Locally stored model
    MODEL_REGISTRY = "registry"  # Centralized model registry
    API = "api"               # External API or service
    CUSTOM = "custom"         # Custom model source


class ModelFramework(Enum):
    """Supported ML frameworks"""
    SCIKIT_LEARN = "sklearn"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    XGB = "xgboost"
    LIGHTGBM = "lightgbm"
    CUSTOM = "custom"


@dataclass
class ModelConfiguration:
    """Configuration for a machine learning model"""
    id: str
    name: str
    description: str
    source: ModelSource
    framework: ModelFramework
    model_path: str  # Path or identifier for the model
    version: str = "1.0.0"
    features: List[str] = field(default_factory=list)
    prediction_type: Optional[PredictionType] = None
    horizon: Optional[PredictionHorizon] = None
    instruments: List[str] = field(default_factory=list)  # Which instruments this model is for
    timeframes: List[str] = field(default_factory=list)   # Which timeframes this model is for
    preprocessing_steps: List[Dict[str, Any]] = field(default_factory=list)
    api_config: Dict[str, Any] = field(default_factory=dict)  # For API-based models
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
    values: Dict[str, Any]  # The actual prediction values
    confidence: float  # 0.0-1.0
    features_used: Dict[str, Any]  # The feature values used for this prediction
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with defaults if needed"""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    @property
    def expiration(self) -> datetime:
        """Get the expiration time of this prediction"""
        # Convert the horizon to timedelta
        horizon_seconds = self.horizon.value
        return self.timestamp + timedelta(seconds=horizon_seconds)
    
    @property
    def is_expired(self) -> bool:
        """Check if the prediction has expired"""
        return datetime.now() > self.expiration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "prediction_type": self.prediction_type.name,
            "horizon": self.horizon.name,
            "values": self.values,
            "confidence": self.confidence,
            "features_used": self.features_used,
            "expires_at": self.expiration.isoformat(),
            "metadata": self.metadata
        }


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
    
    def __init__(self, model_configs: List[ModelConfiguration] = None):
        """
        Initialize the indicator-ML bridge
        
        Args:
            model_configs: List of model configurations to load
        """
        self._models: Dict[str, Any] = {}  # Loaded model objects
        self._model_configs: Dict[str, ModelConfiguration] = {}  # Model configurations
        self._predictions: List[ModelPrediction] = []  # Recent predictions
        
        # Initialize with provided configs
        if model_configs:
            for config in model_configs:
                self.add_model_config(config)
    
    def add_model_config(self, config: ModelConfiguration) -> None:
        """
        Add a model configuration
        
        Args:
            config: The model configuration to add
        """
        self._model_configs[config.id] = config
        logger.info(f"Added model configuration: {config.name} ({config.id})")
    
    def load_model_configs(self, config_path: str) -> int:
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
                    # Convert string enums to actual enum values
                    config_data['source'] = ModelSource(config_data.get('source', 'local'))
                    config_data['framework'] = ModelFramework(config_data.get('framework', 'sklearn'))
                    
                    if 'prediction_type' in config_data:
                        config_data['prediction_type'] = PredictionType[config_data['prediction_type']]
                    
                    if 'horizon' in config_data:
                        config_data['horizon'] = PredictionHorizon[config_data['horizon']]
                    
                    # Create the configuration
                    config = ModelConfiguration(**config_data)
                    self.add_model_config(config)
                    count += 1
                    
                except Exception as e:
                    logger.error(f"Error loading model configuration: {str(e)}")
                    continue
            
            logger.info(f"Loaded {count} model configurations from {config_path}")
            return count
            
        except Exception as e:
            logger.error(f"Error loading model configurations from {config_path}: {str(e)}")
            return 0
    
    def save_model_configs(self, config_path: str) -> bool:
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
                config_dict = {
                    "id": config.id,
                    "name": config.name,
                    "description": config.description,
                    "source": config.source.value,
                    "framework": config.framework.value,
                    "model_path": config.model_path,
                    "version": config.version,
                    "features": config.features,
                    "prediction_type": config.prediction_type.name if config.prediction_type else None,
                    "horizon": config.horizon.name if config.horizon else None,
                    "instruments": config.instruments,
                    "timeframes": config.timeframes,
                    "preprocessing_steps": config.preprocessing_steps,
                    "api_config": config.api_config,
                    "parameters": config.parameters,
                    "metrics": config.metrics,
                    "enabled": config.enabled
                }
                configs_data.append(config_dict)
            
            with open(config_path, 'w') as f:
                json.dump(configs_data, f, indent=4)
            
            logger.info(f"Saved {len(configs_data)} model configurations to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model configurations to {config_path}: {str(e)}")
            return False
    
    def get_model_config(self, model_id: str) -> Optional[ModelConfiguration]:
        """
        Get a model configuration by ID
        
        Args:
            model_id: The ID of the model configuration
            
        Returns:
            The model configuration, or None if not found
        """
        return self._model_configs.get(model_id)
    
    def load_model(self, model_id: str) -> Any:
        """
        Load a model by its configuration ID
        
        Args:
            model_id: The ID of the model to load
            
        Returns:
            The loaded model object
            
        Raises:
            ModelLoadError: If the model cannot be loaded
        """
        # Check if already loaded
        if model_id in self._models:
            return self._models[model_id]
        
        # Get configuration
        config = self.get_model_config(model_id)
        if not config:
            raise ModelLoadError(f"Model configuration not found: {model_id}")
        
        # Check if enabled
        if not config.enabled:
            raise ModelLoadError(f"Model is disabled: {config.name} ({model_id})")
        
        try:
            # Load based on source and framework
            model = None
            
            if config.source == ModelSource.LOCAL:
                # Load from local storage
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
                    # Custom loading logic would be implemented here
                    # For example, loading from a pickle or custom format
                    raise NotImplementedError("Custom model loading not implemented yet")
                    
                else:
                    raise ModelLoadError(f"Unsupported framework: {config.framework}")
                    
            elif config.source == ModelSource.MODEL_REGISTRY:
                # Implement logic to load from a model registry
                # This would typically involve API calls to a model registry service
                raise NotImplementedError("Model registry loading not implemented yet")
                
            elif config.source == ModelSource.API:
                # For API-based models, we don't actually load a model
                # Instead we just set up the API configuration
                model = {"type": "api_model", "config": config.api_config}
                
            elif config.source == ModelSource.CUSTOM:
                # Custom loading logic
                raise NotImplementedError("Custom model source loading not implemented yet")
                
            else:
                raise ModelLoadError(f"Unsupported model source: {config.source}")
            
            # Store the loaded model
            self._models[model_id] = model
            
            logger.info(f"Loaded model: {config.name} ({model_id})")
            return model
            
        except Exception as e:
            error_message = f"Failed to load model {model_id}: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            raise ModelLoadError(error_message) from e
    
    def prepare_features(self, 
                       data: pd.DataFrame, 
                       model_id: str) -> pd.DataFrame:
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
            # Get model configuration
            config = self.get_model_config(model_id)
            if not config:
                raise FeaturePreparationError(f"Model configuration not found: {model_id}")
            
            # Get required features
            required_features = config.features
            if not required_features:
                raise FeaturePreparationError(f"No features specified for model: {model_id}")
            
            # Calculate indicators if needed
            indicator_data = data.copy()
            
            # Extract all indicator names from feature requirements
            # This assumes feature names are in format "indicator_name.feature_name"
            indicators_needed = set()
            for feature in required_features:
                if "." in feature:
                    indicator_name = feature.split(".")[0]
                    indicators_needed.add(indicator_name)
            
            # Calculate required indicators
            for indicator_name in indicators_needed:
                try:
                    result = indicator_registry.calculate_indicator(
                        indicator_name, data, use_cache=True
                    )
                    # Merge resulting columns into indicator_data
                    for col in result.data.columns:
                        if col not in indicator_data.columns:
                            indicator_data[col] = result.data[col]
                            
                except Exception as e:
                    logger.warning(f"Failed to calculate indicator {indicator_name}: {str(e)}")
                    # Continue with other indicators
            
            # Extract requested features
            if hasattr(feature_store, 'extract_features'):
                # Use feature store if available
                feature_names = [f.split(".")[-1] for f in required_features]
                feature_data = feature_store.extract_features(indicator_data, feature_names)
            else:
                # Manual feature extraction
                feature_data = pd.DataFrame(index=indicator_data.index)
                for feature in required_features:
                    if "." in feature:
                        # Format: indicator_name.column_name
                        indicator_name, column_name = feature.split(".", 1)
                        if column_name in indicator_data.columns:
                            feature_data[feature] = indicator_data[column_name]
                    elif feature in indicator_data.columns:
                        # Direct column name
                        feature_data[feature] = indicator_data[feature]
            
            # Apply preprocessing steps from model config
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
                                feature_data[col] = (feature_data[col] - min_val) / (max_val - min_val)
                                
                elif step_type == 'lag':
                    periods = step.get('periods', 1)
                    feature_data = feature_data.shift(periods)
                    
                # Add other preprocessing steps as needed
            
            # Final check for missing values
            if feature_data.isna().any().any():
                # Handle remaining NaNs
                feature_data = feature_data.fillna(0)
                logger.warning(f"Filled NaN values in features for model {model_id}")
            
            return feature_data
            
        except Exception as e:
            error_message = f"Failed to prepare features for model {model_id}: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            raise FeaturePreparationError(error_message) from e
    
    def predict(self, 
              data: pd.DataFrame, 
              model_id: str, 
              instrument: str, 
              timeframe: str) -> ModelPrediction:
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
            # Get model configuration
            config = self.get_model_config(model_id)
            if not config:
                raise PredictionError(f"Model configuration not found: {model_id}")
            
            # Check instrument and timeframe compatibility
            if config.instruments and instrument not in config.instruments:
                raise PredictionError(f"Model {model_id} not configured for instrument {instrument}")
                
            if config.timeframes and timeframe not in config.timeframes:
                raise PredictionError(f"Model {model_id} not configured for timeframe {timeframe}")
            
            # Prepare features
            features = self.prepare_features(data, model_id)
            
            # Load the model
            model = self.load_model(model_id)
            
            # Generate prediction based on model framework
            prediction_values = {}
            confidence = 0.5  # Default confidence
            
            # Get the latest feature values for prediction
            latest_features = features.iloc[-1].to_dict()
            
            # For demonstration, store which feature values were used
            features_used = latest_features.copy()
            
            if config.source == ModelSource.API:
                # Make API call for prediction
                api_config = config.api_config
                url = api_config.get('url')
                
                if not url:
                    raise PredictionError("API URL not configured")
                
                import requests
                
                # Prepare request data
                request_data = {
                    "features": latest_features,
                    "instrument": instrument,
                    "timeframe": timeframe
                }
                
                # Add any additional API config parameters
                headers = api_config.get('headers', {})
                
                # Make the request
                response = requests.post(url, json=request_data, headers=headers)
                
                if response.status_code == 200:
                    result = response.json()
                    prediction_values = result.get('prediction', {})
                    confidence = result.get('confidence', 0.5)
                else:
                    raise PredictionError(f"API request failed: {response.status_code}")
                    
            else:
                # Local model prediction
                
                # Convert features to the right format
                feature_array = features.iloc[-1:].values
                
                if config.framework == ModelFramework.SCIKIT_LEARN:
                    # For sklearn models
                    if hasattr(model, 'predict_proba'):
                        # Classification with probabilities
                        probas = model.predict_proba(feature_array)[0]
                        pred_class = model.classes_[np.argmax(probas)]
                        confidence = np.max(probas)
                        
                        # Store prediction values
                        prediction_values = {
                            'class': str(pred_class),
                            'probability': float(confidence)
                        }
                        
                        # For binary classification, add direction
                        if len(model.classes_) == 2:
                            prediction_values['direction'] = 1 if pred_class == 1 else -1
                            
                    else:
                        # Regression or classification without probabilities
                        pred_value = model.predict(feature_array)[0]
                        
                        if config.prediction_type == PredictionType.PRICE_DIRECTION:
                            prediction_values = {
                                'direction': 1 if pred_value > 0 else (-1 if pred_value < 0 else 0),
                                'value': float(pred_value)
                            }
                        else:
                            prediction_values = {'value': float(pred_value)}
                            
                elif config.framework == ModelFramework.TENSORFLOW:
                    # For TensorFlow models
                    pred_array = model.predict(feature_array)
                    
                    if len(pred_array.shape) > 1 and pred_array.shape[1] > 1:
                        # Multi-class classification
                        pred_class = np.argmax(pred_array[0])
                        confidence = pred_array[0][pred_class]
                        
                        prediction_values = {
                            'class': int(pred_class),
                            'probability': float(confidence)
                        }
                    else:
                        # Regression or binary classification
                        pred_value = float(pred_array[0][0])
                        
                        if config.prediction_type == PredictionType.PRICE_DIRECTION:
                            threshold = config.parameters.get('threshold', 0.5)
                            direction = 1 if pred_value > threshold else (-1 if pred_value < (1 - threshold) else 0)
                            
                            prediction_values = {
                                'direction': direction,
                                'probability': pred_value
                            }
                        else:
                            prediction_values = {'value': pred_value}
                            
                elif config.framework == ModelFramework.PYTORCH:
                    # For PyTorch models
                    import torch
                    
                    # Convert features to tensor
                    X_tensor = torch.tensor(feature_array, dtype=torch.float32)
                    
                    # Set model to evaluation mode
                    model.eval()
                    
                    # Generate prediction
                    with torch.no_grad():
                        pred_tensor = model(X_tensor)
                        
                    # Convert to numpy for easier handling
                    pred_array = pred_tensor.numpy()
                    
                    # Process based on prediction type
                    if config.prediction_type == PredictionType.PRICE_DIRECTION:
                        if pred_array.shape[1] > 1:
                            # Multi-class output
                            pred_class = np.argmax(pred_array[0])
                            confidence = pred_array[0][pred_class]
                            
                            # Map classes to directions (-1, 0, 1)
                            direction_map = {0: -1, 1: 0, 2: 1}  # Assuming 3 classes
                            direction = direction_map.get(pred_class, 0)
                            
                            prediction_values = {
                                'direction': direction,
                                'class': int(pred_class),
                                'probability': float(confidence)
                            }
                        else:
                            # Binary output
                            pred_value = float(pred_array[0][0])
                            threshold = config.parameters.get('threshold', 0.5)
                            direction = 1 if pred_value > threshold else -1
                            
                            prediction_values = {
                                'direction': direction,
                                'probability': pred_value
                            }
                    else:
                        # Regression
                        prediction_values = {'value': float(pred_array[0][0])}
                        
                elif config.framework in (ModelFramework.XGB, ModelFramework.LIGHTGBM):
                    # For XGBoost or LightGBM models
                    if config.framework == ModelFramework.XGB:
                        import xgboost as xgb
                        dmatrix = xgb.DMatrix(feature_array)
                        pred_array = model.predict(dmatrix)
                    else:
                        pred_array = model.predict(feature_array)
                    
                    # Process prediction based on type
                    if len(pred_array.shape) > 1 and pred_array.shape[1] > 1:
                        # Multi-class
                        pred_class = np.argmax(pred_array[0])
                        confidence = pred_array[0][pred_class]
                        
                        prediction_values = {
                            'class': int(pred_class),
                            'probability': float(confidence)
                        }
                    else:
                        # Binary or regression
                        pred_value = float(pred_array[0])
                        
                        if config.prediction_type == PredictionType.PRICE_DIRECTION:
                            threshold = config.parameters.get('threshold', 0.5)
                            direction = 1 if pred_value > threshold else (-1 if pred_value < (1 - threshold) else 0)
                            
                            prediction_values = {
                                'direction': direction,
                                'probability': pred_value
                            }
                        else:
                            prediction_values = {'value': pred_value}
            
            # Create the prediction object
            prediction = ModelPrediction(
                id=str(uuid.uuid4()),
                model_id=model_id,
                model_name=config.name,
                timestamp=datetime.now(),
                instrument=instrument,
                timeframe=timeframe,
                prediction_type=config.prediction_type or PredictionType.PRICE_DIRECTION,
                horizon=config.horizon or PredictionHorizon.HOUR_1,
                values=prediction_values,
                confidence=confidence,
                features_used=features_used,
                metadata={
                    "feature_count": len(features.columns)
                }
            )
            
            # Store the prediction
            self._predictions.append(prediction)
            
            # Trim predictions list if getting too large
            if len(self._predictions) > 1000:
                self._predictions = self._predictions[-1000:]
            
            return prediction
            
        except Exception as e:
            error_message = f"Prediction failed for model {model_id}: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            raise PredictionError(error_message) from e
    
    def predict_multiple(self, 
                        data: pd.DataFrame, 
                        model_ids: List[str], 
                        instrument: str, 
                        timeframe: str,
                        parallel: bool = True) -> List[ModelPrediction]:
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
            # Parallel prediction with ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                futures = []
                
                for model_id in model_ids:
                    future = executor.submit(
                        self.predict, data, model_id, instrument, timeframe
                    )
                    futures.append((model_id, future))
                
                # Collect results
                for model_id, future in futures:
                    try:
                        prediction = future.result()
                        predictions.append(prediction)
                    except Exception as e:
                        logger.error(f"Prediction failed for model {model_id}: {str(e)}")
                        # Continue with other models
        else:
            # Sequential prediction
            for model_id in model_ids:
                try:
                    prediction = self.predict(data, model_id, instrument, timeframe)
                    predictions.append(prediction)
                except Exception as e:
                    logger.error(f"Prediction failed for model {model_id}: {str(e)}")
                    # Continue with other models
        
        return predictions
    
    def get_recent_predictions(self, 
                             instrument: Optional[str] = None, 
                             timeframe: Optional[str] = None,
                             model_id: Optional[str] = None,
                             prediction_type: Optional[PredictionType] = None,
                             max_age: Optional[timedelta] = None) -> List[ModelPrediction]:
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
        
        # Apply filters
        if instrument:
            predictions = [p for p in predictions if p.instrument == instrument]
        
        if timeframe:
            predictions = [p for p in predictions if p.timeframe == timeframe]
        
        if model_id:
            predictions = [p for p in predictions if p.model_id == model_id]
        
        if prediction_type:
            predictions = [p for p in predictions if p.prediction_type == prediction_type]
        
        if max_age:
            min_timestamp = datetime.now() - max_age
            predictions = [p for p in predictions if p.timestamp >= min_timestamp]
        
        # Sort by timestamp (newest first)
        return sorted(predictions, key=lambda p: p.timestamp, reverse=True)
    
    def generate_signals_from_predictions(self, 
                                       predictions: List[ModelPrediction],
                                       confidence_threshold: float = 0.6,
                                       use_signal_system: bool = True) -> List[Dict[str, Any]]:
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
            # Skip expired or low-confidence predictions
            if pred.is_expired or pred.confidence < confidence_threshold:
                continue
            
            # For price direction predictions
            if pred.prediction_type == PredictionType.PRICE_DIRECTION:
                direction = pred.values.get('direction')
                
                if direction == 1:  # Bullish
                    signal_type = SignalType.BUY
                elif direction == -1:  # Bearish
                    signal_type = SignalType.SELL
                else:
                    continue  # No clear signal
                    
                # Create signal
                if use_signal_system:
                    # Use the signal system if available
                    from analysis_engine.analysis.signal_system import Signal
                    
                    signal = Signal(
                        timestamp=pred.timestamp,
                        indicator_name=f"ML:{pred.model_name}",
                        signal_type=signal_type,
                        strength=pred.confidence,
                        price=pred.features_used.get('close', 0.0),
                        metadata={
                            "model_id": pred.model_id,
                            "prediction_horizon": pred.horizon.label,
                            "prediction_id": pred.id
                        }
                    )
                    
                    signal_system.add_signal(signal)
                    
                    signals.append({
                        "id": str(uuid.uuid4()),
                        "prediction_id": pred.id,
                        "model_id": pred.model_id,
                        "instrument": pred.instrument,
                        "timeframe": pred.timeframe,
                        "signal_type": signal_type.name,
                        "strength": pred.confidence,
                        "timestamp": pred.timestamp.isoformat()
                    })
                else:
                    # Just return the signal information
                    signals.append({
                        "id": str(uuid.uuid4()),
                        "prediction_id": pred.id,
                        "model_id": pred.model_id,
                        "instrument": pred.instrument,
                        "timeframe": pred.timeframe,
                        "signal_type": "BUY" if direction == 1 else "SELL",
                        "strength": pred.confidence,
                        "timestamp": pred.timestamp.isoformat()
                    })
        
        return signals
    
    def save_predictions_to_file(self, filepath: str) -> bool:
        """
        Save recent predictions to a JSON file
        
        Args:
            filepath: Path to save the predictions
            
        Returns:
            True if successful, False otherwise
        """
        try:
            predictions_data = [p.to_dict() for p in self._predictions]
            
            with open(filepath, 'w') as f:
                json.dump(predictions_data, f, indent=4)
                
            logger.info(f"Saved {len(predictions_data)} predictions to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save predictions to {filepath}: {str(e)}")
            return False


# Create a global instance for easy import
ml_bridge = IndicatorMLBridge()
