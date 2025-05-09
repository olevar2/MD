"""
ML Prediction Interfaces Module

This module provides interfaces for ML prediction functionality used across services,
helping to break circular dependencies between services.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass


class ModelType(str, Enum):
    """Types of ML models"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    CUSTOM = "custom"


@dataclass
class ModelMetadata:
    """Metadata about an ML model"""
    model_id: str
    model_type: ModelType
    version: str
    created_at: datetime
    features: List[str]
    target: str
    description: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    tags: Optional[List[str]] = None


@dataclass
class PredictionRequest:
    """Request for model prediction"""
    model_id: str
    inputs: Dict[str, Any]
    version_id: Optional[str] = None
    explanation_required: bool = False
    context: Optional[Dict[str, Any]] = None


@dataclass
class PredictionResult:
    """Result of model prediction"""
    prediction: Any
    confidence: float
    model_id: str
    version_id: str
    timestamp: datetime
    explanation: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class IMLPredictionService(ABC):
    """Interface for ML prediction services"""
    
    @abstractmethod
    async def get_prediction(
        self,
        model_id: str,
        inputs: Dict[str, Any],
        version_id: Optional[str] = None,
        explanation_required: bool = False,
        context: Optional[Dict[str, Any]] = None
    ) -> PredictionResult:
        """
        Get a prediction from a model.
        
        Args:
            model_id: Model identifier
            inputs: Input data for the model
            version_id: Optional version identifier
            explanation_required: Whether to include explanation
            context: Optional context information
            
        Returns:
            Prediction result
        """
        pass
    
    @abstractmethod
    async def get_batch_predictions(
        self,
        model_id: str,
        batch_inputs: List[Dict[str, Any]],
        version_id: Optional[str] = None,
        explanation_required: bool = False,
        context: Optional[Dict[str, Any]] = None
    ) -> List[PredictionResult]:
        """
        Get predictions for a batch of inputs.
        
        Args:
            model_id: Model identifier
            batch_inputs: List of input data
            version_id: Optional version identifier
            explanation_required: Whether to include explanations
            context: Optional context information
            
        Returns:
            List of prediction results
        """
        pass
    
    @abstractmethod
    async def get_model_metadata(
        self,
        model_id: str,
        version_id: Optional[str] = None
    ) -> ModelMetadata:
        """
        Get metadata for a model.
        
        Args:
            model_id: Model identifier
            version_id: Optional version identifier
            
        Returns:
            Model metadata
        """
        pass
    
    @abstractmethod
    async def list_available_models(
        self,
        model_type: Optional[ModelType] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelMetadata]:
        """
        List available models.
        
        Args:
            model_type: Optional filter by model type
            tags: Optional filter by tags
            
        Returns:
            List of model metadata
        """
        pass


class IMLSignalGenerator(ABC):
    """Interface for ML signal generation"""
    
    @abstractmethod
    async def generate_trading_signals(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int = 100,
        models: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate trading signals using ML models.
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            lookback_bars: Number of bars to analyze
            models: Optional list of model IDs to use
            context: Optional context information
            
        Returns:
            Dictionary with signals and metadata
        """
        pass
    
    @abstractmethod
    async def get_model_performance(
        self,
        model_id: str,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get performance metrics for a model.
        
        Args:
            model_id: Model identifier
            symbol: Optional symbol filter
            timeframe: Optional timeframe filter
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            Dictionary with performance metrics
        """
        pass
