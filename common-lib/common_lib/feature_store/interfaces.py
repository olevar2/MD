"""
Feature Store Interfaces

This module defines interfaces for feature store providers and consumers
to break circular dependencies between services.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
import pandas as pd
from datetime import datetime

class FeatureType(str, Enum):
    """Enum for feature types."""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    MARKET_REGIME = "market_regime"
    VOLATILITY = "volatility"
    CUSTOM = "custom"

class FeatureScope(str, Enum):
    """Enum for feature scope."""
    SYMBOL = "symbol"
    MARKET = "market"
    SECTOR = "sector"
    GLOBAL = "global"

class FeatureStatus(str, Enum):
    """Enum for feature status."""
    AVAILABLE = "available"
    COMPUTING = "computing"
    ERROR = "error"
    UNAVAILABLE = "unavailable"

class FeatureMetadata(Dict[str, Any]):
    """Type for feature metadata."""
    pass

class IFeatureProvider(ABC):
    """Interface for feature providers."""
    
    @abstractmethod
    async def get_feature(
        self,
        feature_name: str,
        symbol: str,
        timeframe: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        parameters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Get feature data.
        
        Args:
            feature_name: Name of the feature
            symbol: Symbol to get data for
            timeframe: Timeframe to get data for
            start_date: Start date for data
            end_date: End date for data
            parameters: Optional parameters for feature calculation
            
        Returns:
            DataFrame with feature data
        """
        pass
    
    @abstractmethod
    async def compute_feature(
        self,
        feature_name: str,
        symbol: str,
        timeframe: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        parameters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Compute feature data.
        
        Args:
            feature_name: Name of the feature
            symbol: Symbol to compute data for
            timeframe: Timeframe to compute data for
            start_date: Start date for data
            end_date: End date for data
            parameters: Optional parameters for feature calculation
            
        Returns:
            DataFrame with computed feature data
        """
        pass
    
    @abstractmethod
    async def get_available_features(
        self,
        feature_type: Optional[FeatureType] = None,
        scope: Optional[FeatureScope] = None
    ) -> List[Dict[str, Any]]:
        """
        Get available features.
        
        Args:
            feature_type: Optional filter by feature type
            scope: Optional filter by feature scope
            
        Returns:
            List of feature metadata
        """
        pass
    
    @abstractmethod
    async def get_feature_metadata(
        self,
        feature_name: str
    ) -> FeatureMetadata:
        """
        Get feature metadata.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Feature metadata
        """
        pass
    
    @abstractmethod
    async def batch_get_features(
        self,
        features: List[Dict[str, Any]],
        symbol: str,
        timeframe: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> Dict[str, pd.DataFrame]:
        """
        Get multiple features in batch.
        
        Args:
            features: List of feature specifications
            symbol: Symbol to get data for
            timeframe: Timeframe to get data for
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary mapping feature names to DataFrames
        """
        pass

class IFeatureConsumer(ABC):
    """Interface for feature consumers."""
    
    @abstractmethod
    async def consume_feature(
        self,
        feature_name: str,
        data: pd.DataFrame,
        metadata: Optional[FeatureMetadata] = None
    ) -> None:
        """
        Consume feature data.
        
        Args:
            feature_name: Name of the feature
            data: Feature data
            metadata: Optional feature metadata
        """
        pass
    
    @abstractmethod
    async def batch_consume_features(
        self,
        features: Dict[str, pd.DataFrame],
        metadata: Optional[Dict[str, FeatureMetadata]] = None
    ) -> None:
        """
        Consume multiple features in batch.
        
        Args:
            features: Dictionary mapping feature names to DataFrames
            metadata: Optional dictionary mapping feature names to metadata
        """
        pass
    
    @abstractmethod
    async def get_feature_requirements(self) -> List[Dict[str, Any]]:
        """
        Get feature requirements.
        
        Returns:
            List of feature requirements
        """
        pass

class IFeatureStore(ABC):
    """Interface for feature store."""
    
    @abstractmethod
    async def store_feature(
        self,
        feature_name: str,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
        metadata: Optional[FeatureMetadata] = None
    ) -> None:
        """
        Store feature data.
        
        Args:
            feature_name: Name of the feature
            symbol: Symbol the data is for
            timeframe: Timeframe the data is for
            data: Feature data
            metadata: Optional feature metadata
        """
        pass
    
    @abstractmethod
    async def retrieve_feature(
        self,
        feature_name: str,
        symbol: str,
        timeframe: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> pd.DataFrame:
        """
        Retrieve feature data.
        
        Args:
            feature_name: Name of the feature
            symbol: Symbol to retrieve data for
            timeframe: Timeframe to retrieve data for
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with feature data
        """
        pass
    
    @abstractmethod
    async def delete_feature(
        self,
        feature_name: str,
        symbol: str,
        timeframe: str
    ) -> None:
        """
        Delete feature data.
        
        Args:
            feature_name: Name of the feature
            symbol: Symbol to delete data for
            timeframe: Timeframe to delete data for
        """
        pass
    
    @abstractmethod
    async def list_features(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        feature_type: Optional[FeatureType] = None
    ) -> List[Dict[str, Any]]:
        """
        List available features.
        
        Args:
            symbol: Optional filter by symbol
            timeframe: Optional filter by timeframe
            feature_type: Optional filter by feature type
            
        Returns:
            List of feature metadata
        """
        pass

class IFeatureGenerator(ABC):
    """Interface for feature generators."""
    
    @abstractmethod
    async def generate_feature(
        self,
        feature_name: str,
        symbol: str,
        timeframe: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        parameters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Generate feature data.
        
        Args:
            feature_name: Name of the feature
            symbol: Symbol to generate data for
            timeframe: Timeframe to generate data for
            start_date: Start date for data
            end_date: End date for data
            parameters: Optional parameters for feature generation
            
        Returns:
            DataFrame with generated feature data
        """
        pass
    
    @abstractmethod
    async def get_supported_features(self) -> List[Dict[str, Any]]:
        """
        Get supported features.
        
        Returns:
            List of supported feature metadata
        """
        pass
    
    @abstractmethod
    async def validate_parameters(
        self,
        feature_name: str,
        parameters: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate parameters for feature generation.
        
        Args:
            feature_name: Name of the feature
            parameters: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
