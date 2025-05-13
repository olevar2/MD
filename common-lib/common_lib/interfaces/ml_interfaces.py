"""
ML Service Interfaces

This module defines interfaces for ML services used by the analysis engine.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


class IModelProvider(ABC):
    """Interface for model providers."""
    
    @abstractmethod
    async def get_model_prediction(self, model_id: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get prediction from a model.
        
        Args:
            model_id: ID of the model to use
            features: Features to use for prediction
            
        Returns:
            Dictionary of prediction results
        """
        pass


class IModelRegistry(ABC):
    """Interface for model registry services."""
    
    @abstractmethod
    async def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Get metadata for a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary of model metadata
        """
        pass
