"""
ML Integration Service: Prediction Model Integration

This module connects technical indicators and extracted features to machine learning models,
enabling predictive analytics and providing feedback on indicator performance.
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum, auto
from dataclasses import dataclass
import os
from datetime import datetime, timedelta
import time
import uuid
from pathlib import Path

from ml_integration_service.caching.model_inference_cache import cache_model_inference

# Configure logging
logger = logging.getLogger(__name__)


class PredictionHorizon(Enum):
    """Timeframes for predictions"""
    MINUTE_1 = 60           # 1 minute
    MINUTE_5 = 300          # 5 minutes
    MINUTE_15 = 900         # 15 minutes
    MINUTE_30 = 1800        # 30 minutes
    HOUR_1 = 3600           # 1 hour
    HOUR_4 = 14400          # 4 hours
    DAY_1 = 86400           # 1 day
    DAY_3 = 259200          # 3 days
    WEEK_1 = 604800         # 1 week
    MONTH_1 = 2592000       # 1 month (30 days)

    @classmethod
    def from_seconds(cls, seconds: int) -> 'PredictionHorizon':
        """Get closest prediction horizon from seconds"""
        horizons = list(cls)
        closest = min(horizons, key=lambda x: abs(x.value - seconds))
        return closest

    @property
    def label(self) -> str:
        """Get human-readable label"""
        if self == self.MINUTE_1:
            return "1 Minute"
        elif self == self.MINUTE_5:
            return "5 Minutes"
        elif self == self.MINUTE_15:
            return "15 Minutes"
        elif self == self.MINUTE_30:
            return "30 Minutes"
        elif self == self.HOUR_1:
            return "1 Hour"
        elif self == self.HOUR_4:
            return "4 Hours"
        elif self == self.DAY_1:
            return "1 Day"
        elif self == self.DAY_3:
            return "3 Days"
        elif self == self.WEEK_1:
            return "1 Week"
        elif self == self.MONTH_1:
            return "1 Month"
        return str(self)


class PredictionType(Enum):
    """Types of predictions"""
    PRICE_DIRECTION = auto()    # Direction of price movement (up/down)
    PRICE_TARGET = auto()       # Specific price target
    VOLATILITY = auto()         # Expected volatility
    REVERSAL_PROBABILITY = auto() # Probability of trend reversal
    MOMENTUM = auto()           # Momentum strength
    RANGE_BOUND = auto()        # Whether price will stay in a range
    SUPPORT_RESISTANCE = auto() # Support/resistance identification


@dataclass
class ModelMetadata:
    """Metadata for a machine learning model"""
    id: str
    name: str
    description: str
    version: str
    created_at: datetime
    prediction_type: PredictionType
    horizon: PredictionHorizon
    features: List[str]
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    instrument: Optional[str] = None
    timeframe: Optional[str] = None


@dataclass
class PredictionResult:
    """Result of a model prediction"""
    model_id: str
    timestamp: datetime
    prediction_type: PredictionType
    horizon: PredictionHorizon
    value: Any
    probability: float
    confidence_interval: Optional[Tuple[float, float]] = None
    feature_importance: Optional[Dict[str, float]] = None

    @property
    def target_time(self) -> datetime:
        """Get the target time for the prediction"""
        return self.timestamp + timedelta(seconds=self.horizon.value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "model_id": self.model_id,
            "timestamp": self.timestamp.isoformat(),
            "prediction_type": self.prediction_type.name,
            "horizon": self.horizon.name,
            "horizon_seconds": self.horizon.value,
            "value": self.value,
            "probability": self.probability,
            "target_time": self.target_time.isoformat()
        }

        if self.confidence_interval:
            result["confidence_interval"] = self.confidence_interval

        if self.feature_importance:
            result["feature_importance"] = self.feature_importance

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionResult':
        """Create from dictionary"""
        confidence_interval = data.get("confidence_interval")
        feature_importance = data.get("feature_importance")

        return cls(
            model_id=data["model_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            prediction_type=PredictionType[data["prediction_type"]],
            horizon=PredictionHorizon[data["horizon"]],
            value=data["value"],
            probability=data["probability"],
            confidence_interval=confidence_interval,
            feature_importance=feature_importance
        )


class ModelRegistry:
    """Registry for machine learning models"""

    def __init__(self, models_dir: str = "./models"):
        """
        Initialize the model registry

        Args:
            models_dir: Directory to store models
        """
        self.models_dir = models_dir
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, ModelMetadata] = {}

        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)

        # Load model metadata
        self._load_metadata()

    def register_model(self, model: Any, metadata: ModelMetadata) -> str:
        """
        Register a model with the registry

        Args:
            model: The model object
            metadata: Metadata for the model

        Returns:
            Model ID
        """
        # Generate ID if not provided
        if not metadata.id:
            metadata.id = str(uuid.uuid4())

        # Store model and metadata
        self.models[metadata.id] = model
        self.metadata[metadata.id] = metadata

        # Save model and metadata
        self._save_model(metadata.id, model)
        self._save_metadata(metadata.id, metadata)

        logger.info(f"Registered model {metadata.name} (ID: {metadata.id})")

        return metadata.id

    def get_model(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        """
        Get a model by ID

        Args:
            model_id: ID of the model

        Returns:
            Tuple of (model, metadata)
        """
        # Load model if not in memory
        if model_id not in self.models:
            self._load_model(model_id)

        return self.models[model_id], self.metadata[model_id]

    def list_models(self, prediction_type: Optional[PredictionType] = None,
                  instrument: Optional[str] = None) -> List[ModelMetadata]:
        """
        List available models with optional filtering

        Args:
            prediction_type: Filter by prediction type
            instrument: Filter by instrument

        Returns:
            List of model metadata
        """
        results = list(self.metadata.values())

        # Apply filters
        if prediction_type:
            results = [m for m in results if m.prediction_type == prediction_type]

        if instrument:
            results = [m for m in results
                     if m.instrument is None or m.instrument == instrument]

        return results

    def _model_path(self, model_id: str) -> str:
        """Get the path to a model file"""
        return os.path.join(self.models_dir, f"{model_id}.joblib")

    def _metadata_path(self, model_id: str) -> str:
        """Get the path to a metadata file"""
        return os.path.join(self.models_dir, f"{model_id}.json")

    def _save_model(self, model_id: str, model: Any) -> None:
        """Save a model to disk"""
        model_path = self._model_path(model_id)
        joblib.dump(model, model_path)

    def _save_metadata(self, model_id: str, metadata: ModelMetadata) -> None:
        """Save model metadata to disk"""
        metadata_path = self._metadata_path(model_id)

        # Convert to dictionary
        data = {
            "id": metadata.id,
            "name": metadata.name,
            "description": metadata.description,
            "version": metadata.version,
            "created_at": metadata.created_at.isoformat(),
            "prediction_type": metadata.prediction_type.name,
            "horizon": metadata.horizon.name,
            "features": metadata.features,
            "metrics": metadata.metrics,
            "parameters": metadata.parameters,
            "instrument": metadata.instrument,
            "timeframe": metadata.timeframe
        }

        # Save to file
        with open(metadata_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_model(self, model_id: str) -> None:
        """Load a model from disk"""
        model_path = self._model_path(model_id)

        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                self.models[model_id] = model
                logger.debug(f"Loaded model {model_id} from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model {model_id}: {str(e)}")
                raise
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    def _load_metadata(self) -> None:
        """Load all model metadata from disk"""
        # Get all JSON files in the models directory
        metadata_files = Path(self.models_dir).glob("*.json")

        for metadata_path in metadata_files:
            model_id = metadata_path.stem

            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)

                # Create ModelMetadata object
                metadata = ModelMetadata(
                    id=data["id"],
                    name=data["name"],
                    description=data["description"],
                    version=data["version"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    prediction_type=PredictionType[data["prediction_type"]],
                    horizon=PredictionHorizon[data["horizon"]],
                    features=data["features"],
                    metrics=data["metrics"],
                    parameters=data["parameters"],
                    instrument=data.get("instrument"),
                    timeframe=data.get("timeframe")
                )

                # Store metadata
                self.metadata[model_id] = metadata
                logger.debug(f"Loaded metadata for model {model_id}")

            except Exception as e:
                logger.error(f"Error loading metadata for {model_id}: {str(e)}")


class FeedbackCollector:
    """Collects feedback on model and indicator performance"""

    def __init__(self, feedback_dir: str = "./feedback"):
        """
        Initialize the feedback collector

        Args:
            feedback_dir: Directory to store feedback
        """
        self.feedback_dir = feedback_dir

        # Create feedback directory if it doesn't exist
        os.makedirs(feedback_dir, exist_ok=True)

        # Store active predictions
        self.active_predictions: Dict[str, PredictionResult] = {}

        # Store indicator-model linkage
        self.indicator_model_usage: Dict[str, Set[str]] = {}

    def record_prediction(self, prediction: PredictionResult) -> str:
        """
        Record a new prediction for later evaluation

        Args:
            prediction: The prediction result

        Returns:
            Prediction ID
        """
        # Generate unique ID for the prediction
        prediction_id = str(uuid.uuid4())

        # Store prediction
        self.active_predictions[prediction_id] = prediction

        # Store prediction to disk
        self._save_prediction(prediction_id, prediction)

        return prediction_id

    def record_feedback(self, prediction_id: str, actual_value: Any,
                      metrics: Dict[str, float]) -> None:
        """
        Record feedback for a prediction

        Args:
            prediction_id: ID of the prediction
            actual_value: The actual observed value
            metrics: Dictionary of evaluation metrics
        """
        if prediction_id not in self.active_predictions:
            raise ValueError(f"Unknown prediction ID: {prediction_id}")

        prediction = self.active_predictions[prediction_id]

        # Create feedback record
        feedback = {
            "prediction_id": prediction_id,
            "model_id": prediction.model_id,
            "prediction": prediction.to_dict(),
            "actual_value": actual_value,
            "metrics": metrics,
            "feedback_time": datetime.now().isoformat()
        }

        # Save feedback to disk
        self._save_feedback(prediction_id, feedback)

        # Remove from active predictions
        del self.active_predictions[prediction_id]

        logger.debug(f"Recorded feedback for prediction {prediction_id}")

    def register_indicator_usage(self, model_id: str, indicator_names: List[str]) -> None:
        """
        Register which indicators are used by a model

        Args:
            model_id: ID of the model
            indicator_names: List of indicator names used by the model
        """
        for indicator in indicator_names:
            if indicator not in self.indicator_model_usage:
                self.indicator_model_usage[indicator] = set()
            self.indicator_model_usage[indicator].add(model_id)

    def get_indicator_performance(self, indicator_name: str) -> Dict[str, Any]:
        """
        Get performance metrics for an indicator based on model performance

        Args:
            indicator_name: Name of the indicator

        Returns:
            Dictionary with performance metrics
        """
        # Get models using this indicator
        if indicator_name not in self.indicator_model_usage:
            return {"models_count": 0}

        model_ids = self.indicator_model_usage[indicator_name]

        # Collect feedback for these models
        model_metrics = {}
        for model_id in model_ids:
            metrics = self._get_model_metrics(model_id)
            if metrics:
                model_metrics[model_id] = metrics

        # Aggregate metrics across models
        if not model_metrics:
            return {"models_count": len(model_ids)}

        # Calculate average metrics
        avg_metrics = {}
        for metric in next(iter(model_metrics.values())).keys():
            values = [m[metric] for m in model_metrics.values() if metric in m]
            if values:
                avg_metrics[metric] = sum(values) / len(values)

        return {
            "models_count": len(model_ids),
            "models_with_metrics": len(model_metrics),
            "avg_metrics": avg_metrics
        }

    def _get_model_metrics(self, model_id: str) -> Dict[str, float]:
        """Get aggregated metrics for a model"""
        # This would query the feedback database or files
        # For now, we'll return a simple placeholder
        return {}

    def _prediction_path(self, prediction_id: str) -> str:
        """Get the path to a prediction file"""
        return os.path.join(self.feedback_dir, f"prediction_{prediction_id}.json")

    def _feedback_path(self, prediction_id: str) -> str:
        """Get the path to a feedback file"""
        return os.path.join(self.feedback_dir, f"feedback_{prediction_id}.json")

    def _save_prediction(self, prediction_id: str, prediction: PredictionResult) -> None:
        """Save a prediction to disk"""
        prediction_path = self._prediction_path(prediction_id)

        with open(prediction_path, 'w') as f:
            json.dump(prediction.to_dict(), f, indent=2)

    def _save_feedback(self, prediction_id: str, feedback: Dict[str, Any]) -> None:
        """Save feedback to disk"""
        feedback_path = self._feedback_path(prediction_id)

        with open(feedback_path, 'w') as f:
            json.dump(feedback, f, indent=2)


class ModelConnector:
    """Connects indicators and features to machine learning models"""

    def __init__(self, registry: Optional[ModelRegistry] = None,
               feedback: Optional[FeedbackCollector] = None):
        """
        Initialize the model connector

        Args:
            registry: Model registry (created if None)
            feedback: Feedback collector (created if None)
        """
        self.registry = registry or ModelRegistry()
        self.feedback = feedback or FeedbackCollector()

        # Cache for feature data
        self.feature_cache = {}

    @cache_model_inference(ttl=1800)  # Cache for 30 minutes
    def predict(self, model_id: str, symbol: str, timeframe: str, features: pd.DataFrame) -> PredictionResult:
        """
        Make a prediction using a model

        Args:
            model_id: ID of the model to use
            symbol: Trading symbol
            timeframe: Chart timeframe
            features: DataFrame with feature data

        Returns:
            PredictionResult with the prediction
        """
        # Get model and metadata
        model, metadata = self.registry.get_model(model_id)

        # Check that all required features are present
        missing_features = [f for f in metadata.features if f not in features.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Extract only needed features in correct order
        X = features[metadata.features]

        # Make prediction
        start_time = time.time()

        try:
            # For classification models
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X)
                y_pred = model.predict(X)

                if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                    # Multi-class or binary classification
                    probability = np.max(y_prob, axis=1)[0]
                else:
                    # Single class probability
                    probability = y_prob[0][0]

                value = y_pred[0]

            # For regression models
            else:
                y_pred = model.predict(X)
                value = y_pred[0]
                probability = 1.0  # No probability for regression

            # Get feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_importance = dict(zip(metadata.features, importance))
            elif hasattr(model, 'coef_'):
                importance = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                feature_importance = dict(zip(metadata.features, importance))

            # Create prediction result
            result = PredictionResult(
                model_id=model_id,
                timestamp=datetime.now(),
                prediction_type=metadata.prediction_type,
                horizon=metadata.horizon,
                value=value,
                probability=probability,
                feature_importance=feature_importance
            )

            # Record prediction for feedback
            prediction_id = self.feedback.record_prediction(result)

            # Store prediction ID in the result's metadata for reference
            result.prediction_id = prediction_id

            return result

        except Exception as e:
            logger.error(f"Error making prediction with model {model_id}: {str(e)}")
            raise

        finally:
            prediction_time = time.time() - start_time
            logger.debug(f"Prediction with model {model_id} took {prediction_time:.3f}s")

    def update_feature_data(self, indicator_name: str, data: pd.DataFrame) -> None:
        """
        Update feature data from an indicator

        Args:
            indicator_name: Name of the indicator
            data: DataFrame with indicator data
        """
        # Store in cache
        self.feature_cache[indicator_name] = data

    def evaluate_prediction(self, prediction_id: str, actual_value: Any) -> Dict[str, float]:
        """
        Evaluate a prediction against actual value and record feedback

        Args:
            prediction_id: ID of the prediction to evaluate
            actual_value: The actual observed value

        Returns:
            Dictionary of evaluation metrics
        """
        if prediction_id not in self.feedback.active_predictions:
            raise ValueError(f"Unknown prediction ID: {prediction_id}")

        prediction = self.feedback.active_predictions[prediction_id]

        # Calculate metrics based on prediction type
        metrics = {}

        if prediction.prediction_type == PredictionType.PRICE_DIRECTION:
            # For directional predictions (up/down)
            correct = (prediction.value > 0 and actual_value > 0) or \
                     (prediction.value < 0 and actual_value < 0)
            metrics["accuracy"] = 1.0 if correct else 0.0

        elif prediction.prediction_type == PredictionType.PRICE_TARGET:
            # For price target predictions
            error = abs(prediction.value - actual_value)
            metrics["absolute_error"] = error
            metrics["percentage_error"] = error / actual_value if actual_value != 0 else float('inf')

        elif prediction.prediction_type == PredictionType.VOLATILITY:
            # For volatility predictions
            error = abs(prediction.value - actual_value)
            metrics["volatility_error"] = error

        # Record feedback
        self.feedback.record_feedback(prediction_id, actual_value, metrics)

        return metrics


def implement_prediction_integration():
    """
    Connects indicators/features to machine learning models.
    - Implements feedback mechanism for indicator improvement.
    - Develops interface for future trend prediction.

    Returns:
        The model connector instance
    """
    # Initialize model registry with path
    registry = ModelRegistry(models_dir="./models")

    # Initialize feedback collector
    feedback = FeedbackCollector(feedback_dir="./feedback")

    # Create model connector
    connector = ModelConnector(registry=registry, feedback=feedback)

    logger.info("Prediction model integration initialized")

    return connector
