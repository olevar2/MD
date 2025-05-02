"""
Transfer Learning Module

This module provides transfer learning capabilities for financial models,
enabling knowledge transfer between instruments and timeframes.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json

logger = logging.getLogger(__name__)


class ModelFeatureTransformer:
    """
    Transforms features between different instruments or timeframes
    to enable transfer learning.
    """
    
    def __init__(self):
        """Initialize the feature transformer"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.transformation_cache = {}
    
    def fit_transform_mapping(
        self,
        source_features: pd.DataFrame,
        target_features: pd.DataFrame,
        feature_subset: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Learn a transformation mapping from source to target features.
        
        Args:
            source_features: Features from the source domain (e.g., EURUSD)
            target_features: Features from the target domain (e.g., GBPUSD)
            feature_subset: Optional subset of features to transform
            
        Returns:
            Transformation parameters
        """
        if source_features.empty or target_features.empty:
            self.logger.warning("Empty features provided for mapping")
            return {}
        
        # Ensure indices are aligned
        source_features = source_features.copy()
        target_features = target_features.copy()
        
        # Use only specified features if provided
        if feature_subset:
            source_cols = [col for col in feature_subset if col in source_features.columns]
            target_cols = [col for col in feature_subset if col in target_features.columns]
            common_cols = list(set(source_cols).intersection(target_cols))
            
            if not common_cols:
                self.logger.warning("No common features found in subset")
                return {}
                
            source_features = source_features[common_cols]
            target_features = target_features[common_cols]
        else:
            # Find common features
            common_cols = list(set(source_features.columns).intersection(target_features.columns))
            source_features = source_features[common_cols]
            target_features = target_features[common_cols]
        
        # Calculate transformation parameters for each feature
        transformation = {
            "features": {},
            "metadata": {
                "source_shape": source_features.shape,
                "target_shape": target_features.shape,
                "timestamp": datetime.now().isoformat(),
                "common_features": common_cols
            }
        }
        
        for col in common_cols:
            if source_features[col].std() == 0 or target_features[col].std() == 0:
                # Skip features with no variance
                continue
                
            # Calculate simple linear transformation parameters
            # target = a * source + b
            s_mean = source_features[col].mean()
            s_std = source_features[col].std()
            t_mean = target_features[col].mean()
            t_std = target_features[col].std()
            
            # Calculate scaling factor and offset
            a = t_std / s_std if s_std > 0 else 1.0
            b = t_mean - a * s_mean
            
            # Store parameters
            transformation["features"][col] = {
                "scale": a,
                "offset": b,
                "source_mean": s_mean,
                "source_std": s_std,
                "target_mean": t_mean,
                "target_std": t_std
            }
        
        return transformation
    
    def transform_features(
        self,
        features: pd.DataFrame,
        transformation: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Apply transformation to source features to map them to target domain.
        
        Args:
            features: Source features to transform
            transformation: Transformation parameters from fit_transform_mapping
            
        Returns:
            Transformed features
        """
        if not transformation or "features" not in transformation:
            self.logger.warning("Invalid transformation provided")
            return features
        
        result = features.copy()
        
        for col, params in transformation["features"].items():
            if col in result.columns:
                # Apply linear transformation
                result[col] = params["scale"] * result[col] + params["offset"]
        
        return result
    
    def get_transfer_similarity(
        self,
        source_features: pd.DataFrame,
        target_features: pd.DataFrame,
        feature_subset: Optional[List[str]] = None
    ) -> float:
        """
        Calculate how similar two sets of features are for transfer learning.
        Higher similarity score means better transfer learning potential.
        
        Args:
            source_features: Features from source domain
            target_features: Features from target domain
            feature_subset: Optional subset of features to consider
            
        Returns:
            Similarity score (0-1)
        """
        # Use only specified features if provided
        if feature_subset:
            source_cols = [col for col in feature_subset if col in source_features.columns]
            target_cols = [col for col in feature_subset if col in target_features.columns]
            common_cols = list(set(source_cols).intersection(target_cols))
        else:
            # Find common features
            common_cols = list(set(source_features.columns).intersection(target_features.columns))
        
        if not common_cols:
            return 0.0
        
        source_subset = source_features[common_cols]
        target_subset = target_features[common_cols]
        
        # Calculate correlation matrix between source and target
        correlations = []
        for col in common_cols:
            corr = np.corrcoef(
                source_subset[col].values,
                target_subset[col].values
            )[0, 1]
            
            if not np.isnan(corr):
                correlations.append(abs(corr))
        
        if not correlations:
            return 0.0
            
        # Average absolute correlation as similarity score
        return sum(correlations) / len(correlations)


class TransferLearningModel:
    """
    Base class for models that support transfer learning.
    
    This class enables transferring knowledge between:
    1. Different instruments/assets (e.g., EURUSD -> GBPUSD)
    2. Different timeframes (e.g., 1h -> 15m)
    3. Different market regimes (e.g., trending -> ranging)
    """
    
    def __init__(self, 
                 source_model_path: Optional[str] = None,
                 feature_transformer: Optional[ModelFeatureTransformer] = None):
        """
        Initialize transfer learning model
        
        Args:
            source_model_path: Path to source model weights/parameters
            feature_transformer: Transformer for feature adaptation
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.source_model_path = source_model_path
        self.feature_transformer = feature_transformer or ModelFeatureTransformer()
        self.transfer_params = {}
        self.model_params = {}
        self.is_transfer_model = False
        
    def load_source_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load source model for transfer learning
        
        Args:
            model_path: Path to model file
            
        Returns:
            Success flag
        """
        path = model_path or self.source_model_path
        if not path or not os.path.exists(path):
            self.logger.error(f"Source model path does not exist: {path}")
            return False
            
        try:
            # Implementation depends on model type (could be scikit-learn, PyTorch, TensorFlow)
            # For this POC, we'll implement a simple JSON parameter loading
            with open(path, 'r') as f:
                params = json.load(f)
                
            self.model_params = params
            self.is_transfer_model = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load source model: {e}")
            return False
    
    def adapt_layers(self, 
                     source_data: pd.DataFrame,
                     target_data: pd.DataFrame,
                     layers_to_adapt: List[str] = None) -> Dict[str, Any]:
        """
        Adapt specific layers from source to target domain
        
        Args:
            source_data: Sample data from source domain
            target_data: Sample data from target domain
            layers_to_adapt: Names of layers to adapt
            
        Returns:
            Parameters for layer adaptation
        """
        # This is a simplified implementation for the POC
        # A real implementation would adapt network layers or model parameters
        
        # For the POC, we'll focus on feature transformations
        feature_mapping = self.feature_transformer.fit_transform_mapping(
            source_data, target_data)
        
        # Calculate similarity score
        similarity = self.feature_transformer.get_transfer_similarity(
            source_data, target_data)
            
        adaptation_params = {
            "feature_mapping": feature_mapping,
            "similarity_score": similarity,
            "layers_adapted": layers_to_adapt or [],
            "timestamp": datetime.now().isoformat()
        }
        
        self.transfer_params = adaptation_params
        return adaptation_params
    
    def save_transfer_model(self, path: str) -> bool:
        """
        Save the transfer-adapted model
        
        Args:
            path: Path to save the model
            
        Returns:
            Success flag
        """
        try:
            # Create combined parameters
            params = {
                "model_params": self.model_params,
                "transfer_params": self.transfer_params,
                "is_transfer_model": True,
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "source_model": self.source_model_path
                }
            }
            
            # Save as JSON (simplified for POC)
            with open(path, 'w') as f:
                json.dump(params, f, indent=2)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save transfer model: {e}")
            return False
    
    def transform_input_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input features to match the source model's expected format
        
        Args:
            features: Input features from target domain
            
        Returns:
            Transformed features
        """
        if not self.is_transfer_model or not self.transfer_params:
            return features
            
        feature_mapping = self.transfer_params.get("feature_mapping", {})
        if not feature_mapping:
            return features
            
        return self.feature_transformer.transform_features(features, feature_mapping)


class TransferLearningFactory:
    """
    Factory for creating and managing transfer learning models.
    
    This class provides methods to:
    1. Discover potential transfer learning opportunities
    2. Create transfer learning models
    3. Evaluate transfer learning effectiveness
    """
    
    def __init__(self):
        """Initialize the transfer learning factory"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.feature_transformer = ModelFeatureTransformer()
    
    def find_transfer_candidates(
        self,
        target_symbol: str,
        target_timeframe: str,
        source_data: Dict[str, pd.DataFrame],
        min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find suitable source models for transfer learning
        
        Args:
            target_symbol: Target instrument
            target_timeframe: Target timeframe
            source_data: Dictionary of available source data by symbol/timeframe
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of transfer candidates with similarity scores
        """
        candidates = []
        target_key = f"{target_symbol}_{target_timeframe}"
        
        if target_key not in source_data:
            self.logger.warning(f"No target data found for {target_key}")
            return candidates
            
        target_features = source_data[target_key]
        
        # Compare with each source
        for source_key, source_features in source_data.items():
            if source_key == target_key:
                continue
                
            # Calculate similarity score
            similarity = self.feature_transformer.get_transfer_similarity(
                source_features, target_features)
                
            # If above threshold, add as candidate
            if similarity >= min_similarity:
                source_parts = source_key.split('_')
                candidates.append({
                    "source_symbol": source_parts[0],
                    "source_timeframe": source_parts[1],
                    "similarity": similarity,
                    "key": source_key
                })
        
        # Sort by similarity (descending)
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates
    
    def create_transfer_model(
        self,
        source_model_path: str,
        source_data: pd.DataFrame,
        target_data: pd.DataFrame,
        adapt_layers: List[str] = None,
        output_path: Optional[str] = None
    ) -> Tuple[TransferLearningModel, Dict[str, Any]]:
        """
        Create a transfer learning model from source to target
        
        Args:
            source_model_path: Path to source model
            source_data: Sample data from source domain
            target_data: Sample data from target domain
            adapt_layers: List of layers to adapt
            output_path: Optional path to save the transfer model
            
        Returns:
            Transfer model and adaptation metrics
        """
        # Create transfer model
        model = TransferLearningModel(
            source_model_path=source_model_path,
            feature_transformer=self.feature_transformer
        )
        
        # Load source model
        if not model.load_source_model():
            self.logger.error("Failed to load source model")
            return model, {"success": False, "error": "Failed to load source model"}
        
        # Adapt layers
        adaptation_params = model.adapt_layers(
            source_data, target_data, adapt_layers)
            
        # Save if path provided
        if output_path:
            model.save_transfer_model(output_path)
            
        metrics = {
            "success": True,
            "similarity": adaptation_params.get("similarity_score", 0),
            "adapted_features": len(adaptation_params.get("feature_mapping", {}).get("features", {})),
            "source_model": source_model_path
        }
        
        return model, metrics
    
    def evaluate_transfer_effectiveness(
        self,
        transfer_model: TransferLearningModel,
        target_data: pd.DataFrame,
        target_labels: pd.Series
    ) -> Dict[str, Any]:
        """
        Evaluate how well the transfer learning worked
        
        Args:
            transfer_model: Transfer learning model
            target_data: Evaluation data from target domain
            target_labels: Ground truth labels for target data
            
        Returns:
            Evaluation metrics
        """
        # This is a simplified implementation
        # In a real system, this would perform model evaluation
        
        # Transform features
        transformed_data = transfer_model.transform_input_features(target_data)
        
        # In a real implementation, we would:
        # 1. Use the transfer model to make predictions
        # 2. Calculate metrics (accuracy, F1, RMSE, etc.)
        # 3. Compare with baseline without transfer learning
        
        # For the POC, return placeholder metrics
        return {
            "transfer_similarity": transfer_model.transfer_params.get("similarity_score", 0),
            "features_adapted": len(transfer_model.transfer_params.get("feature_mapping", {}).get("features", {})),
            "data_shape_before": target_data.shape,
            "data_shape_after": transformed_data.shape
        }
