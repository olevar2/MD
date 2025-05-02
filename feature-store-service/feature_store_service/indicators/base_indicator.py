"""
Base Indicator Module.

This module provides the base class for all technical indicators in the feature store.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd


class BaseIndicator(ABC):
    """
    Base class for all technical indicators.
    
    This abstract class defines the interface that all technical indicators
    must implement. It provides common functionality and enforces a consistent
    API across all indicator implementations.
    """
    
    # Class-level attributes that can be overridden by subclasses
    category: str = "base"       # Category of the indicator (e.g., "trend", "momentum", "volatility")
    default_params: Dict[str, Any] = {}  # Default parameters for the indicator
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicator values for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicator value columns
            
        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement calculate() method")
    
    def validate_input(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that the input data contains all required columns.
        
        Args:
            data: DataFrame to validate
            required_columns: List of column names that must be present
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If any required column is missing
        """
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        return True
    
    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """
        Get metadata about this indicator class.
        
        Returns:
            Dictionary with indicator metadata
        """
        return {
            "name": cls.__name__,
            "category": getattr(cls, "category", "base"),
            "description": cls.__doc__.strip() if cls.__doc__ else "No description available",
            "default_params": getattr(cls, "default_params", {})
        }
    
    def update_parameters(self, **kwargs) -> None:
        """
        Update indicator parameters after initialization.
        
        Args:
            **kwargs: New parameter values
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter for {self.__class__.__name__}: {key}")
    
    def __str__(self) -> str:
        """String representation of the indicator."""
        params = {
            key: getattr(self, key) 
            for key in self.default_params.keys() 
            if hasattr(self, key)
        }
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in params.items())})"
