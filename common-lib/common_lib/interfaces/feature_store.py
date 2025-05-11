
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
import pandas as pd
from datetime import datetime

class IFeatureProvider(ABC):
    """Interface for feature providers"""

    @abstractmethod
    async def get_feature(self,
                         feature_name: str,
                         symbol: str,
                         timeframe: str,
                         start_time: datetime,
                         end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Get a specific feature for a symbol"""
        pass

    @abstractmethod
    async def get_features(self,
                          feature_names: List[str],
                          symbol: str,
                          timeframe: str,
                          start_time: datetime,
                          end_time: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """Get multiple features for a symbol"""
        pass

    @abstractmethod
    async def list_features(self) -> List[Dict[str, Any]]:
        """List available features"""
        pass

class IFeatureStore(ABC):
    """Interface for feature storage"""

    @abstractmethod
    async def store_feature(self,
                           feature_name: str,
                           symbol: str,
                           timeframe: str,
                           data: pd.DataFrame) -> bool:
        """Store a feature in the feature store"""
        pass

    @abstractmethod
    async def get_feature(self,
                         feature_name: str,
                         symbol: str,
                         timeframe: str,
                         start_time: datetime,
                         end_time: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Retrieve a feature from the feature store"""
        pass

    @abstractmethod
    async def delete_feature(self,
                            feature_name: str,
                            symbol: Optional[str] = None,
                            timeframe: Optional[str] = None) -> bool:
        """Delete a feature from the feature store"""
        pass

class IFeatureGenerator(ABC):
    """Interface for feature generators"""

    @abstractmethod
    async def generate_feature(self,
                              feature_name: str,
                              symbol: str,
                              timeframe: str,
                              start_time: datetime,
                              end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Generate a feature for a symbol"""
        pass

    @abstractmethod
    async def register_generator(self,
                                feature_name: str,
                                generator_func: Callable) -> bool:
        """Register a new feature generator function"""
        pass

    @abstractmethod
    async def list_generators(self) -> List[str]:
        """List available feature generators"""
        pass
