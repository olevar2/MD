"""
ML Adapters Module

This module provides adapter implementations for ML interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging
import asyncio
import json

from common_lib.ml.interfaces import (
    ModelType, ModelFramework, IMLModelProvider, IRLModelTrainer
)
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class MLModelProviderAdapter(IMLModelProvider):
    """
    Adapter for ML model providers that implements the common interface.
    
    This adapter can either wrap an actual provider instance or provide
    standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, provider_instance=None):
        """
        Initialize the adapter.
        
        Args:
            provider_instance: Optional actual provider instance to wrap
        """
        self.provider = provider_instance
        self.model_cache = {}
        self.prediction_history = []
    
    async def get_model_prediction(
        self,
        model_id: str,
        features: Dict[str, Any],
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get prediction from a machine learning model.
        
        Args:
            model_id: ID of the model to use
            features: Input features for prediction
            version: Optional model version
            
        Returns:
            Dictionary with prediction results
        """
        if self.provider:
            try:
                # Try to use the wrapped provider if available
                return await self.provider.get_model_prediction(
                    model_id=model_id,
                    features=features,
                    version=version
                )
            except Exception as e:
                logger.warning(f"Error getting model prediction: {str(e)}")
        
        # Fallback implementation
        logger.info(f"Using fallback prediction for model {model_id}")
        
        # Generate a simple fallback prediction
        prediction = {
            "model_id": model_id,
            "version": version or "latest",
            "timestamp": datetime.now().isoformat(),
            "prediction": None,
            "confidence": 0.5,
            "is_fallback": True
        }
        
        # Different fallback logic based on model type
        if "classification" in model_id.lower():
            prediction["prediction"] = "neutral"
            prediction["probabilities"] = {"bullish": 0.3, "neutral": 0.4, "bearish": 0.3}
        elif "regression" in model_id.lower():
            prediction["prediction"] = 0.0
            prediction["range"] = [-0.5, 0.5]
        elif "regime" in model_id.lower():
            prediction["prediction"] = "ranging_narrow"
            prediction["probabilities"] = {
                "trending_bullish": 0.2,
                "trending_bearish": 0.2,
                "ranging_narrow": 0.3,
                "ranging_wide": 0.2,
                "volatile": 0.1
            }
        
        # Store in history
        self.prediction_history.append({
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "features": features,
            "prediction": prediction
        })
        
        return prediction
    
    async def get_model_metadata(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get metadata about a machine learning model.
        
        Args:
            model_id: ID of the model
            version: Optional model version
            
        Returns:
            Dictionary with model metadata
        """
        if self.provider:
            try:
                # Try to use the wrapped provider if available
                return await self.provider.get_model_metadata(
                    model_id=model_id,
                    version=version
                )
            except Exception as e:
                logger.warning(f"Error getting model metadata: {str(e)}")
        
        # Fallback implementation
        logger.info(f"Using fallback metadata for model {model_id}")
        
        # Generate fallback metadata
        metadata = {
            "model_id": model_id,
            "version": version or "latest",
            "created_at": datetime.now().isoformat(),
            "model_type": ModelType.CLASSIFICATION,
            "framework": ModelFramework.SCIKIT_LEARN,
            "features": ["price_change", "volume", "volatility", "trend_strength"],
            "performance": {
                "accuracy": 0.75,
                "precision": 0.7,
                "recall": 0.8,
                "f1_score": 0.75
            },
            "is_fallback": True
        }
        
        return metadata
    
    async def get_feature_importance(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get feature importance for a model.
        
        Args:
            model_id: ID of the model
            version: Optional model version
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.provider:
            try:
                # Try to use the wrapped provider if available
                return await self.provider.get_feature_importance(
                    model_id=model_id,
                    version=version
                )
            except Exception as e:
                logger.warning(f"Error getting feature importance: {str(e)}")
        
        # Fallback implementation
        logger.info(f"Using fallback feature importance for model {model_id}")
        
        # Generate fallback feature importance
        return {
            "price_change": 0.35,
            "volume": 0.25,
            "volatility": 0.20,
            "trend_strength": 0.15,
            "support_resistance": 0.05
        }
