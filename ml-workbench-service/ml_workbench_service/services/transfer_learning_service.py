"""
Transfer Learning Service

This module provides service-level functionality for transfer learning,
serving as a layer between API endpoints and the underlying model components.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
from datetime import datetime
import json

from ml_workbench_service.transfer_learning.model import (
    TransferLearningFactory, 
    TransferLearningModel, 
    ModelFeatureTransformer
)

logger = logging.getLogger(__name__)

class TransferLearningService:
    """
    Service for managing transfer learning operations.
    
    This service provides methods to:
    1. Find transfer learning opportunities
    2. Create and manage transfer learning models
    3. Evaluate transfer learning effectiveness
    4. Apply transfer learning to new data
    """
    
    def __init__(self, 
                 model_registry_path: str = None,
                 output_path: str = None):
        """
        Initialize the transfer learning service
        
        Args:
            model_registry_path: Path to the model registry
            output_path: Path for storing transfer models
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Set default paths if not provided
        if not model_registry_path:
            self.model_registry_path = os.path.join(
                os.getcwd(), 
                "models"
            )
        else:
            self.model_registry_path = model_registry_path
            
        if not output_path:
            self.output_path = os.path.join(
                os.getcwd(), 
                "models", 
                "transfer_models"
            )
        else:
            self.output_path = output_path
            
        # Ensure directories exist
        os.makedirs(self.model_registry_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        
        # Create factory and transformer instances
        self.factory = TransferLearningFactory()
        self.feature_transformer = ModelFeatureTransformer()
        
        # Cache for available models
        self._available_models_cache = {}
        self._last_cache_update = None
        
    def refresh_model_cache(self, force: bool = False) -> None:
        """
        Refresh the cache of available models
        
        Args:
            force: Force refresh even if cache is recent
        """
        # Check if cache needs refreshing
        now = datetime.now()
        if not force and self._last_cache_update and \
           (now - self._last_cache_update).total_seconds() < 300:  # 5 minute cache
            return
            
        try:
            models = {}
            
            # Scan model registry for available models
            for root, dirs, files in os.walk(self.model_registry_path):
                for file in files:
                    if file.endswith(".json"):  # Model metadata files
                        try:
                            model_path = os.path.join(root, file)
                            with open(model_path, 'r') as f:
                                metadata = json.load(f)
                                
                            # Extract key information
                            model_id = metadata.get("model_id", os.path.splitext(file)[0])
                            models[model_id] = {
                                "id": model_id,
                                "path": model_path,
                                "symbol": metadata.get("symbol"),
                                "timeframe": metadata.get("timeframe"),
                                "created_at": metadata.get("created_at"),
                                "metrics": metadata.get("metrics", {}),
                                "is_transfer_model": metadata.get("is_transfer_model", False),
                                "source_model": metadata.get("source_model") if metadata.get("is_transfer_model") else None
                            }
                        except Exception as e:
                            self.logger.warning(f"Error loading model metadata from {file}: {e}")
            
            self._available_models_cache = models
            self._last_cache_update = now
            
        except Exception as e:
            self.logger.error(f"Error refreshing model cache: {e}")
    
    def get_available_models(self, 
                           symbol: Optional[str] = None,
                           timeframe: Optional[str] = None,
                           is_transfer_model: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Get a list of available models with optional filtering
        
        Args:
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            is_transfer_model: Filter by transfer model status
            
        Returns:
            List of model information dictionaries
        """
        # Refresh cache if needed
        self.refresh_model_cache()
        
        # Apply filters
        result = list(self._available_models_cache.values())
        
        if symbol:
            result = [model for model in result if model.get("symbol") == symbol]
            
        if timeframe:
            result = [model for model in result if model.get("timeframe") == timeframe]
            
        if is_transfer_model is not None:
            result = [model for model in result if model.get("is_transfer_model") == is_transfer_model]
            
        # Sort by creation date (newest first)
        result.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return result
    
    def find_transfer_candidates(self,
                               target_symbol: str,
                               target_timeframe: str,
                               source_data: Dict[str, pd.DataFrame],
                               min_similarity: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find suitable source models for transfer learning
        
        Args:
            target_symbol: Target instrument
            target_timeframe: Target timeframe
            source_data: Dictionary of available data by instrument/timeframe
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of transfer candidates with similarity scores and metadata
        """
        # Use factory to find candidates
        candidates = self.factory.find_transfer_candidates(
            target_symbol,
            target_timeframe,
            source_data,
            min_similarity
        )
        
        # Enhance with available model information
        self.refresh_model_cache()
        
        enhanced_candidates = []
        for candidate in candidates:
            source_symbol = candidate.get("source_symbol")
            source_timeframe = candidate.get("source_timeframe")
            
            # Find matching models in cache
            matching_models = [
                model for model in self._available_models_cache.values()
                if model.get("symbol") == source_symbol and 
                model.get("timeframe") == source_timeframe and
                not model.get("is_transfer_model", False)  # Prefer base models over transfer models
            ]
            
            # Sort by creation date (newest first)
            matching_models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            if matching_models:
                # Add model information to candidate
                candidate["models"] = matching_models
                enhanced_candidates.append(candidate)
                
        return enhanced_candidates
    
    def create_transfer_model(self,
                            source_model_id: str,
                            source_data: pd.DataFrame,
                            target_data: pd.DataFrame,
                            target_symbol: str,
                            target_timeframe: str,
                            adapt_layers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a transfer learning model
        
        Args:
            source_model_id: ID of the source model
            source_data: Training data from source domain
            target_data: Training data from target domain
            target_symbol: Target instrument symbol
            target_timeframe: Target timeframe
            adapt_layers: Optional list of layers to adapt
            
        Returns:
            Result dictionary with model information
        """
        # Refresh model cache
        self.refresh_model_cache()
        
        # Check if source model exists
        if source_model_id not in self._available_models_cache:
            return {
                "success": False,
                "error": f"Source model '{source_model_id}' not found"
            }
            
        source_model_info = self._available_models_cache[source_model_id]
        source_model_path = source_model_info.get("path")
        
        if not os.path.exists(source_model_path):
            return {
                "success": False,
                "error": f"Source model file not found at {source_model_path}"
            }
            
        # Generate output path
        transfer_model_id = f"transfer_{source_model_id}_to_{target_symbol}_{target_timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_path = os.path.join(self.output_path, f"{transfer_model_id}.json")
        
        # Create transfer model
        try:
            model, metrics = self.factory.create_transfer_model(
                source_model_path=source_model_path,
                source_data=source_data,
                target_data=target_data,
                adapt_layers=adapt_layers,
                output_path=output_path
            )
            
            # Add metadata
            transfer_metadata = {
                "model_id": transfer_model_id,
                "source_model_id": source_model_id,
                "symbol": target_symbol,
                "timeframe": target_timeframe,
                "is_transfer_model": True,
                "created_at": datetime.now().isoformat(),
                "metrics": metrics
            }
            
            # Save metadata
            metadata_path = os.path.join(self.output_path, f"{transfer_model_id}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(transfer_metadata, f, indent=2)
                
            # Update cache with new model
            self.refresh_model_cache(force=True)
            
            # Return success with model info
            return {
                "success": True,
                "model_id": transfer_model_id,
                "metrics": metrics,
                "source_model": source_model_id,
                "path": output_path
            }
            
        except Exception as e:
            self.logger.error(f"Error creating transfer model: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def evaluate_transfer_model(self,
                              model_id: str,
                              test_data: pd.DataFrame,
                              test_labels: pd.Series) -> Dict[str, Any]:
        """
        Evaluate a transfer model on test data
        
        Args:
            model_id: ID of the transfer model
            test_data: Test data for evaluation
            test_labels: Ground truth labels
            
        Returns:
            Evaluation metrics
        """
        # Refresh model cache
        self.refresh_model_cache()
        
        # Check if model exists
        if model_id not in self._available_models_cache:
            return {
                "success": False,
                "error": f"Model '{model_id}' not found"
            }
            
        model_info = self._available_models_cache[model_id]
        model_path = model_info.get("path")
        
        if not os.path.exists(model_path):
            return {
                "success": False,
                "error": f"Model file not found at {model_path}"
            }
            
        # Load model
        try:
            model = TransferLearningModel(source_model_path=model_path)
            if not model.load_source_model():
                return {
                    "success": False,
                    "error": f"Failed to load model from {model_path}"
                }
                
            # Evaluate model
            metrics = self.factory.evaluate_transfer_effectiveness(
                model, test_data, test_labels
            )
            
            # Add success flag
            metrics["success"] = True
            metrics["model_id"] = model_id
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating transfer model: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def transform_features(self,
                         model_id: str,
                         features: pd.DataFrame) -> Dict[str, Any]:
        """
        Transform features using a transfer model
        
        Args:
            model_id: ID of the transfer model
            features: Features to transform
            
        Returns:
            Transformed features and metadata
        """
        # Refresh model cache
        self.refresh_model_cache()
        
        # Check if model exists
        if model_id not in self._available_models_cache:
            return {
                "success": False,
                "error": f"Model '{model_id}' not found"
            }
            
        model_info = self._available_models_cache[model_id]
        model_path = model_info.get("path")
        
        if not os.path.exists(model_path):
            return {
                "success": False,
                "error": f"Model file not found at {model_path}"
            }
            
        # Load model
        try:
            model = TransferLearningModel(source_model_path=model_path)
            if not model.load_source_model():
                return {
                    "success": False,
                    "error": f"Failed to load model from {model_path}"
                }
                
            # Transform features
            transformed = model.transform_input_features(features)
            
            return {
                "success": True,
                "original_shape": features.shape,
                "transformed_shape": transformed.shape,
                "transformed_features": transformed.to_dict(orient="records") if not transformed.empty else []
            }
            
        except Exception as e:
            self.logger.error(f"Error transforming features: {e}")
            return {
                "success": False,
                "error": str(e)
            }
