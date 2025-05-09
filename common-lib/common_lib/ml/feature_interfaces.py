"""
Feature Interfaces Module

This module provides interfaces for feature extraction and ML integration components,
helping to break circular dependencies between services.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd


class FeatureType(str, Enum):
    """Types of features for ML models."""
    RAW = "raw"
    NORMALIZED = "normalized"
    STANDARDIZED = "standardized"
    CATEGORICAL = "categorical"
    TREND = "trend"
    CROSSOVER = "crossover"
    DIVERGENCE = "divergence"
    SEQUENTIAL = "sequential"
    CUSTOM = "custom"


class FeatureScope(str, Enum):
    """Scope of a feature."""
    PRICE = "price"
    VOLUME = "volume"
    INDICATOR = "indicator"
    PATTERN = "pattern"
    SENTIMENT = "sentiment"
    FUNDAMENTAL = "fundamental"
    MARKET_REGIME = "market_regime"
    CUSTOM = "custom"


class SelectionMethod(str, Enum):
    """Methods for feature selection."""
    IMPORTANCE = "importance"
    CORRELATION = "correlation"
    VARIANCE = "variance"
    RECURSIVE = "recursive"
    CUSTOM = "custom"


class IFeatureDefinition(ABC):
    """Interface for feature definitions."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the feature."""
        pass
    
    @property
    @abstractmethod
    def source_columns(self) -> List[str]:
        """Get the source columns for the feature."""
        pass
    
    @property
    @abstractmethod
    def feature_type(self) -> FeatureType:
        """Get the type of the feature."""
        pass
    
    @property
    @abstractmethod
    def params(self) -> Dict[str, Any]:
        """Get the parameters for the feature."""
        pass
    
    @property
    @abstractmethod
    def scope(self) -> FeatureScope:
        """Get the scope of the feature."""
        pass
    
    @property
    @abstractmethod
    def is_sequential(self) -> bool:
        """Check if the feature is sequential."""
        pass
    
    @property
    @abstractmethod
    def lookback_periods(self) -> int:
        """Get the lookback periods for the feature."""
        pass


class IFeatureProvider(ABC):
    """Interface for feature providers."""
    
    @abstractmethod
    def get_available_features(self, symbol: str = None) -> List[str]:
        """
        Get a list of available features.
        
        Args:
            symbol: Optional symbol to get features for
            
        Returns:
            List of feature names
        """
        pass
    
    @abstractmethod
    def get_feature_data(
        self,
        feature_names: List[str],
        symbol: str,
        timeframe: str,
        start_date: str = None,
        end_date: str = None,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Get feature data for a symbol.
        
        Args:
            feature_names: List of feature names to get
            symbol: The trading symbol
            timeframe: The timeframe to get data for
            start_date: Optional start date
            end_date: Optional end date
            limit: Optional limit on number of rows
            
        Returns:
            DataFrame with feature data
        """
        pass
    
    @abstractmethod
    def compute_features(
        self,
        feature_definitions: List[Dict[str, Any]],
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute features from raw data.
        
        Args:
            feature_definitions: List of feature definitions
            data: DataFrame with raw data
            
        Returns:
            DataFrame with computed features
        """
        pass
    
    @abstractmethod
    def get_feature_metadata(self, feature_name: str) -> Dict[str, Any]:
        """
        Get metadata for a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Dictionary with feature metadata
        """
        pass


class IFeatureTransformer(ABC):
    """Interface for feature transformers."""
    
    @abstractmethod
    def transform(
        self,
        data: pd.DataFrame,
        feature_definitions: List[Dict[str, Any]] = None,
        fit: bool = False
    ) -> pd.DataFrame:
        """
        Transform features.
        
        Args:
            data: DataFrame with raw data
            feature_definitions: Optional list of feature definitions
            fit: Whether to fit the transformer
            
        Returns:
            DataFrame with transformed features
        """
        pass
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the transformer to data.
        
        Args:
            data: DataFrame with raw data
        """
        pass
    
    @abstractmethod
    def save_state(self, path: str) -> None:
        """
        Save the transformer state.
        
        Args:
            path: Path to save the state to
        """
        pass
    
    @abstractmethod
    def load_state(self, path: str) -> None:
        """
        Load the transformer state.
        
        Args:
            path: Path to load the state from
        """
        pass


class IFeatureSelector(ABC):
    """Interface for feature selectors."""
    
    @abstractmethod
    def select_features(
        self,
        features: pd.DataFrame,
        target: pd.Series = None,
        method: str = None,
        n_features: int = None
    ) -> pd.DataFrame:
        """
        Select features.
        
        Args:
            features: DataFrame with features
            target: Optional target variable
            method: Optional selection method
            n_features: Optional number of features to select
            
        Returns:
            DataFrame with selected features
        """
        pass
    
    @abstractmethod
    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get feature importances.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    @abstractmethod
    def save_state(self, path: str) -> None:
        """
        Save the selector state.
        
        Args:
            path: Path to save the state to
        """
        pass
    
    @abstractmethod
    def load_state(self, path: str) -> None:
        """
        Load the selector state.
        
        Args:
            path: Path to load the state from
        """
        pass


class IMLFeatureConsumer(ABC):
    """Interface for ML feature consumers."""
    
    @abstractmethod
    def get_required_features(self, model_id: str = None) -> List[Dict[str, Any]]:
        """
        Get required features for a model.
        
        Args:
            model_id: Optional model ID
            
        Returns:
            List of feature definitions
        """
        pass
    
    @abstractmethod
    def prepare_model_inputs(
        self,
        features: pd.DataFrame,
        model_id: str = None,
        target_column: str = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare inputs for a model.
        
        Args:
            features: DataFrame with features
            model_id: Optional model ID
            target_column: Optional target column
            
        Returns:
            Tuple of (X, y) where X is the feature matrix and y is the target
        """
        pass
    
    @abstractmethod
    def get_feature_importance_feedback(
        self,
        model_id: str,
        features: List[str]
    ) -> Dict[str, float]:
        """
        Get feature importance feedback from a model.
        
        Args:
            model_id: Model ID
            features: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
