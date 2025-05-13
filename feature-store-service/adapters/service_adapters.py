"""
Service Adapters Module

This module provides adapter implementations for service interfaces,
helping to break circular dependencies between services.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from common_lib.interfaces.feature_store import IFeatureProvider, IFeatureStore, IFeatureGenerator
from common_lib.interfaces.market_data import IMarketDataProvider
from common_lib.adapters import AdapterFactory


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class FeatureProviderAdapter(IFeatureProvider):
    """
    Adapter implementation for the Feature Provider interface.
    
    This adapter implements the Feature Provider interface using the
    feature store service's internal components.
    """

    def __init__(self, logger: Optional[logging.Logger]=None):
        """
        Initialize the Feature Provider adapter.
        
        Args:
            logger: Logger to use (if None, creates a new logger)
        """
        self.logger = logger or logging.getLogger(
            f'{__name__}.{self.__class__.__name__}')
        from feature_store_service.services.feature_service import FeatureService
        self.feature_service = FeatureService()

    @async_with_exception_handling
    async def get_feature(self, feature_name: str, symbol: str, timeframe:
        str, start_time: datetime, end_time: datetime) ->pd.DataFrame:
        """
        Retrieve a feature for the specified parameters.
        
        Args:
            feature_name: Name of the feature to retrieve
            symbol: Trading symbol (e.g., "EUR/USD")
            timeframe: Timeframe for the data (e.g., "1m", "5m", "1h", "1d")
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            
        Returns:
            DataFrame containing the feature data
        """
        try:
            feature_data = await self.feature_service.get_feature(feature_name
                =feature_name, symbol=symbol, timeframe=timeframe,
                start_time=start_time, end_time=end_time)
            return feature_data
        except Exception as e:
            self.logger.error(f'Error retrieving feature: {str(e)}')
            return pd.DataFrame(columns=[feature_name])

    @async_with_exception_handling
    async def get_features(self, feature_names: List[str], symbol: str,
        timeframe: str, start_time: datetime, end_time: datetime
        ) ->pd.DataFrame:
        """
        Retrieve multiple features for the specified parameters.
        
        Args:
            feature_names: List of feature names to retrieve
            symbol: Trading symbol (e.g., "EUR/USD")
            timeframe: Timeframe for the data (e.g., "1m", "5m", "1h", "1d")
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            
        Returns:
            DataFrame containing the feature data
        """
        try:
            features_data = await self.feature_service.get_features(
                feature_names=feature_names, symbol=symbol, timeframe=
                timeframe, start_time=start_time, end_time=end_time)
            return features_data
        except Exception as e:
            self.logger.error(f'Error retrieving features: {str(e)}')
            return pd.DataFrame(columns=feature_names)

    @async_with_exception_handling
    async def get_available_features(self) ->List[str]:
        """
        Get the list of available features.
        
        Returns:
            List of available feature names
        """
        try:
            available_features = (await self.feature_service.
                get_available_features())
            return available_features
        except Exception as e:
            self.logger.error(f'Error retrieving available features: {str(e)}')
            return []

    @async_with_exception_handling
    async def get_feature_metadata(self, feature_name: str) ->Dict[str, Any]:
        """
        Get metadata about the specified feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Dictionary containing metadata about the feature
        """
        try:
            metadata = await self.feature_service.get_feature_metadata(
                feature_name)
            return metadata
        except Exception as e:
            self.logger.error(f'Error retrieving feature metadata: {str(e)}')
            return {}


class FeatureStoreAdapter(IFeatureStore):
    """
    Adapter implementation for the Feature Store interface.
    
    This adapter implements the Feature Store interface using the
    feature store service's internal components.
    """

    def __init__(self, logger: Optional[logging.Logger]=None):
        """
        Initialize the Feature Store adapter.
        
        Args:
            logger: Logger to use (if None, creates a new logger)
        """
        self.logger = logger or logging.getLogger(
            f'{__name__}.{self.__class__.__name__}')
        from feature_store_service.services.feature_service import FeatureService
        self.feature_service = FeatureService()

    @async_with_exception_handling
    async def store_feature(self, feature_name: str, symbol: str, timeframe:
        str, data: pd.DataFrame, metadata: Optional[Dict[str, Any]]=None
        ) ->bool:
        """
        Store a feature in the feature store.
        
        Args:
            feature_name: Name of the feature to store
            symbol: Trading symbol (e.g., "EUR/USD")
            timeframe: Timeframe for the data (e.g., "1m", "5m", "1h", "1d")
            data: DataFrame containing the feature data
            metadata: Optional metadata about the feature
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            success = await self.feature_service.store_feature(feature_name
                =feature_name, symbol=symbol, timeframe=timeframe, data=
                data, metadata=metadata)
            return success
        except Exception as e:
            self.logger.error(f'Error storing feature: {str(e)}')
            return False

    @async_with_exception_handling
    async def delete_feature(self, feature_name: str, symbol: Optional[str]
        =None, timeframe: Optional[str]=None) ->bool:
        """
        Delete a feature from the feature store.
        
        Args:
            feature_name: Name of the feature to delete
            symbol: Trading symbol to delete, or None to delete all symbols
            timeframe: Timeframe to delete, or None to delete all timeframes
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            success = await self.feature_service.delete_feature(feature_name
                =feature_name, symbol=symbol, timeframe=timeframe)
            return success
        except Exception as e:
            self.logger.error(f'Error deleting feature: {str(e)}')
            return False

    @async_with_exception_handling
    async def update_feature_metadata(self, feature_name: str, metadata:
        Dict[str, Any]) ->bool:
        """
        Update metadata for a feature.
        
        Args:
            feature_name: Name of the feature
            metadata: Updated metadata
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            success = await self.feature_service.update_feature_metadata(
                feature_name=feature_name, metadata=metadata)
            return success
        except Exception as e:
            self.logger.error(f'Error updating feature metadata: {str(e)}')
            return False


class FeatureGeneratorAdapter(IFeatureGenerator):
    """
    Adapter implementation for the Feature Generator interface.
    
    This adapter implements the Feature Generator interface using the
    feature store service's internal components.
    """

    def __init__(self, logger: Optional[logging.Logger]=None):
        """
        Initialize the Feature Generator adapter.
        
        Args:
            logger: Logger to use (if None, creates a new logger)
        """
        self.logger = logger or logging.getLogger(
            f'{__name__}.{self.__class__.__name__}')
        from feature_store_service.services.feature_service import FeatureService
        self.feature_service = FeatureService()

    @async_with_exception_handling
    async def generate_feature(self, feature_name: str, symbol: str,
        timeframe: str, start_time: datetime, end_time: datetime,
        parameters: Optional[Dict[str, Any]]=None) ->pd.DataFrame:
        """
        Generate a feature for the specified parameters.
        
        Args:
            feature_name: Name of the feature to generate
            symbol: Trading symbol (e.g., "EUR/USD")
            timeframe: Timeframe for the data (e.g., "1m", "5m", "1h", "1d")
            start_time: Start time for data generation
            end_time: End time for data generation
            parameters: Optional parameters for feature generation
            
        Returns:
            DataFrame containing the generated feature data
        """
        try:
            feature_data = await self.feature_service.generate_feature(
                feature_name=feature_name, symbol=symbol, timeframe=
                timeframe, start_time=start_time, end_time=end_time,
                parameters=parameters)
            return feature_data
        except Exception as e:
            self.logger.error(f'Error generating feature: {str(e)}')
            return pd.DataFrame(columns=[feature_name])

    @async_with_exception_handling
    async def get_available_generators(self) ->List[str]:
        """
        Get the list of available feature generators.
        
        Returns:
            List of available feature generator names
        """
        try:
            available_generators = (await self.feature_service.
                get_available_generators())
            return available_generators
        except Exception as e:
            self.logger.error(
                f'Error retrieving available generators: {str(e)}')
            return []

    @async_with_exception_handling
    async def get_generator_parameters(self, generator_name: str) ->Dict[
        str, Any]:
        """
        Get the parameters for a feature generator.
        
        Args:
            generator_name: Name of the feature generator
            
        Returns:
            Dictionary containing the parameters for the generator
        """
        try:
            parameters = await self.feature_service.get_generator_parameters(
                generator_name)
            return parameters
        except Exception as e:
            self.logger.error(
                f'Error retrieving generator parameters: {str(e)}')
            return {}
