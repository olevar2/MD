"""
Data Service Interfaces

This module defines interfaces for data services used by the analysis engine.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


class IFeatureProvider(ABC):
    """Interface for feature providers."""
    
    @abstractmethod
    async def get_features(self, feature_names: List[str], start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Get features for a specific time range.
        
        Args:
            feature_names: List of feature names to retrieve
            start_time: Start time for the data
            end_time: End time for the data
            
        Returns:
            Dictionary of features
        """
        pass


class IDataPipeline(ABC):
    """Interface for data pipeline services."""
    
    @abstractmethod
    async def get_market_data(self, symbols: List[str], timeframe: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Get market data for a specific time range.
        
        Args:
            symbols: List of symbols to retrieve data for
            timeframe: Timeframe for the data (e.g., "1m", "5m", "1h")
            start_time: Start time for the data
            end_time: End time for the data
            
        Returns:
            Dictionary of market data
        """
        pass
