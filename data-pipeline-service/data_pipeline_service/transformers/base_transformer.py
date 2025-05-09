"""
Base Market Data Transformer

This module provides the base class for all market data transformers.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseMarketDataTransformer(ABC):
    """
    Abstract base class for all market data transformers.
    
    This class defines the common interface and functionality for transforming
    market data, regardless of asset class or operation type.
    """
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        """
        Initialize the base transformer.
        
        Args:
            name: Name identifier for the transformer
            parameters: Configuration parameters for the transformer
        """
        self.name = name
        self.parameters = parameters or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform market data.
        
        Args:
            data: Market data DataFrame
            **kwargs: Additional arguments for the transformation
            
        Returns:
            Transformed market data
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data before transformation.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            True if data is valid, False otherwise
        """
        # Check if data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            self.logger.error("Input data is not a pandas DataFrame")
            return False
        
        # Check if data is empty
        if data.empty:
            self.logger.warning("Input data is empty")
            return False
        
        # Check for required columns (can be overridden by subclasses)
        required_columns = self.get_required_columns()
        if required_columns:
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
        
        return True
    
    def get_required_columns(self) -> List[str]:
        """
        Get the list of required columns for this transformer.
        
        Returns:
            List of required column names
        """
        # Default implementation - can be overridden by subclasses
        return []
    
    def handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data in the input DataFrame.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            DataFrame with missing data handled
        """
        # Default implementation - can be overridden by subclasses
        return data
    
    def log_transformation_stats(self, original: pd.DataFrame, transformed: pd.DataFrame):
        """
        Log statistics about the transformation.
        
        Args:
            original: Original DataFrame
            transformed: Transformed DataFrame
        """
        self.logger.info(f"Transformed data: {len(original)} rows in, {len(transformed)} rows out")
        
        # Log column changes
        added_columns = [col for col in transformed.columns if col not in original.columns]
        if added_columns:
            self.logger.info(f"Added columns: {added_columns}")
        
        removed_columns = [col for col in original.columns if col not in transformed.columns]
        if removed_columns:
            self.logger.info(f"Removed columns: {removed_columns}")
    
    def execute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Execute the transformation with validation and logging.
        
        Args:
            data: Market data DataFrame
            **kwargs: Additional arguments for the transformation
            
        Returns:
            Transformed market data
        """
        # Validate input data
        if not self.validate_data(data):
            self.logger.warning("Data validation failed, returning original data")
            return data
        
        # Handle missing data
        cleaned_data = self.handle_missing_data(data)
        
        # Perform the transformation
        try:
            transformed_data = self.transform(cleaned_data, **kwargs)
            
            # Log transformation statistics
            self.log_transformation_stats(data, transformed_data)
            
            return transformed_data
        
        except Exception as e:
            self.logger.error(f"Error during transformation: {str(e)}", exc_info=True)
            return data