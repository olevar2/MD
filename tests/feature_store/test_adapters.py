"""
Feature Store Test Adapters

This module provides test adapters for feature store interfaces
to break circular dependencies between feature-store-service and tests.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
from datetime import datetime
import logging

from common_lib.feature_store.interfaces import (
    IFeatureProvider,
    IFeatureStore,
    IFeatureGenerator,
    FeatureType,
    FeatureScope,
    FeatureMetadata
)

# Configure logger
logger = logging.getLogger(__name__)

class TestFeatureProviderAdapter(IFeatureProvider):
    """Test adapter implementation for IFeatureProvider interface."""
    
    def __init__(self, test_data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initialize the adapter.
        
        Args:
            test_data: Optional test data dictionary
        """
        self.test_data = test_data or {}
    
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
        key = f"{feature_name}_{symbol}_{timeframe}"
        if key in self.test_data:
            return self.test_data[key]
        
        # Generate mock data if not available
        return self._generate_mock_data(feature_name, start_date, end_date)
    
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
        # For testing, just return the same as get_feature
        return await self.get_feature(
            feature_name=feature_name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            parameters=parameters
        )
    
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
        # Return mock feature metadata
        features = [
            {
                "name": "sma",
                "type": FeatureType.TECHNICAL,
                "scope": FeatureScope.SYMBOL,
                "parameters": {"period": {"type": "int", "default": 14}}
            },
            {
                "name": "rsi",
                "type": FeatureType.TECHNICAL,
                "scope": FeatureScope.SYMBOL,
                "parameters": {"period": {"type": "int", "default": 14}}
            },
            {
                "name": "volatility",
                "type": FeatureType.VOLATILITY,
                "scope": FeatureScope.SYMBOL,
                "parameters": {"window": {"type": "int", "default": 14}}
            }
        ]
        
        # Apply filters
        if feature_type:
            features = [f for f in features if f["type"] == feature_type]
        
        if scope:
            features = [f for f in features if f["scope"] == scope]
        
        return features
    
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
        # Return mock metadata for the feature
        features = await self.get_available_features()
        for feature in features:
            if feature["name"] == feature_name:
                return feature
        
        return {}
    
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
        result = {}
        
        for feature in features:
            feature_name = feature["name"]
            parameters = feature.get("parameters")
            
            result[feature_name] = await self.get_feature(
                feature_name=feature_name,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                parameters=parameters
            )
        
        return result
    
    def _generate_mock_data(
        self,
        feature_name: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> pd.DataFrame:
        """
        Generate mock data for testing.
        
        Args:
            feature_name: Name of the feature
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with mock data
        """
        import numpy as np
        
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        
        # Generate mock data
        data = pd.DataFrame({
            "timestamp": date_range,
            feature_name: np.random.randn(len(date_range))
        })
        
        return data

class TestFeatureStoreAdapter(IFeatureStore):
    """Test adapter implementation for IFeatureStore interface."""
    
    def __init__(self):
        """Initialize the adapter."""
        self.stored_features = {}
    
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
        key = f"{feature_name}_{symbol}_{timeframe}"
        self.stored_features[key] = {
            "data": data.copy(),
            "metadata": metadata or {}
        }
    
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
        key = f"{feature_name}_{symbol}_{timeframe}"
        if key in self.stored_features:
            data = self.stored_features[key]["data"]
            
            # Filter by date range
            if "timestamp" in data.columns:
                # Convert string dates to datetime if needed
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
                
                return data[(data["timestamp"] >= start_date) & (data["timestamp"] <= end_date)]
            
            return data
        
        return pd.DataFrame()
    
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
        key = f"{feature_name}_{symbol}_{timeframe}"
        if key in self.stored_features:
            del self.stored_features[key]
    
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
        result = []
        
        for key, value in self.stored_features.items():
            feature_name, feature_symbol, feature_timeframe = key.split("_")
            
            # Apply filters
            if symbol and feature_symbol != symbol:
                continue
            
            if timeframe and feature_timeframe != timeframe:
                continue
            
            metadata = value.get("metadata", {})
            if feature_type and metadata.get("type") != feature_type:
                continue
            
            result.append({
                "name": feature_name,
                "symbol": feature_symbol,
                "timeframe": feature_timeframe,
                **metadata
            })
        
        return result
