"""
Data Validator for the ML Integration Service.

This module provides functionality for validating data used by the ML Integration Service.
"""

from typing import Dict, List, Any, Optional
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataValidator:
    """Validator for model data."""
    
    def __init__(self):
        """Initialize the data validator."""
        pass
        
    def validate_training_data(self, data: pd.DataFrame) -> bool:
        """
        Validate training data.
        
        Args:
            data: Training data to validate
            
        Returns:
            Whether the data is valid
        """
        # Check if data is empty
        if data.empty:
            logger.warning("Training data is empty")
            return False
            
        # Check for missing values
        if data.isnull().any().any():
            logger.warning("Training data contains missing values")
            return False
            
        # Check for required columns
        required_columns = ["timestamp", "target"]
        for column in required_columns:
            if column not in data.columns:
                logger.warning(f"Training data is missing required column: {column}")
                return False
                
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            logger.warning("Timestamp column is not a datetime type")
            return False
            
        # Check for duplicate timestamps
        if data["timestamp"].duplicated().any():
            logger.warning("Training data contains duplicate timestamps")
            return False
            
        return True
        
    def validate_inference_data(self, data: pd.DataFrame) -> bool:
        """
        Validate inference data.
        
        Args:
            data: Inference data to validate
            
        Returns:
            Whether the data is valid
        """
        # Check if data is empty
        if data.empty:
            logger.warning("Inference data is empty")
            return False
            
        # Check for missing values
        if data.isnull().any().any():
            logger.warning("Inference data contains missing values")
            return False
            
        # Check for required columns
        required_columns = ["timestamp", "prediction"]
        for column in required_columns:
            if column not in data.columns:
                logger.warning(f"Inference data is missing required column: {column}")
                return False
                
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            logger.warning("Timestamp column is not a datetime type")
            return False
            
        # Check for duplicate timestamps
        if data["timestamp"].duplicated().any():
            logger.warning("Inference data contains duplicate timestamps")
            return False
            
        return True
        
    def validate_feature_data(self, data: pd.DataFrame, required_features: List[str]) -> bool:
        """
        Validate feature data.
        
        Args:
            data: Feature data to validate
            required_features: List of required features
            
        Returns:
            Whether the data is valid
        """
        # Check if data is empty
        if data.empty:
            logger.warning("Feature data is empty")
            return False
            
        # Check for required columns
        for feature in required_features:
            if feature not in data.columns:
                logger.warning(f"Feature data is missing required feature: {feature}")
                return False
                
        # Check for missing values in required features
        for feature in required_features:
            if data[feature].isnull().any():
                logger.warning(f"Feature data contains missing values in feature: {feature}")
                return False
                
        return True
