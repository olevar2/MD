"""
Feature Store Service Adapter.

This module provides adapter implementations for the Feature Store Service interfaces.
These adapters allow other services to interact with the Feature Store Service
without direct dependencies, breaking circular dependencies.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from common_lib.interfaces.feature_store import IFeatureProvider, IFeatureStore, IFeatureGenerator
from common_lib.service_client.base_client import ServiceClientConfig
from common_lib.service_client.http_client import AsyncHTTPServiceClient


class FeatureProviderAdapter(IFeatureProvider):
    """
    Adapter implementation for the Feature Provider interface.
    
    This adapter uses the HTTP service client to communicate with the Feature Store Service.
    """
    
    def __init__(
        self,
        config: ServiceClientConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Feature Provider adapter.
        
        Args:
            config: Configuration for the service client
            logger: Logger to use (if None, creates a new logger)
        """
        self.client = AsyncHTTPServiceClient(config)
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def get_feature(
        self,
        feature_name: str,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
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
            # Prepare request parameters
            params = {
                "feature_name": feature_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            # Send request to the Feature Store Service
            response = await self.client.send_request({
                "method": "GET",
                "path": "api/v1/features",
                "params": params
            })
            
            # Convert response to DataFrame
            if "data" in response and response["data"]:
                df = pd.DataFrame(response["data"])
                
                # Convert timestamp to datetime
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)
                
                return df
            else:
                # Return empty DataFrame with feature name as column
                return pd.DataFrame(columns=[feature_name])
        except Exception as e:
            self.logger.error(f"Error retrieving feature: {str(e)}")
            raise
    
    async def get_features(
        self,
        feature_names: List[str],
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
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
            # Prepare request parameters
            params = {
                "feature_names": ",".join(feature_names),
                "symbol": symbol,
                "timeframe": timeframe,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            # Send request to the Feature Store Service
            response = await self.client.send_request({
                "method": "GET",
                "path": "api/v1/features/batch",
                "params": params
            })
            
            # Convert response to DataFrame
            if "data" in response and response["data"]:
                df = pd.DataFrame(response["data"])
                
                # Convert timestamp to datetime
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)
                
                return df
            else:
                # Return empty DataFrame with feature names as columns
                return pd.DataFrame(columns=feature_names)
        except Exception as e:
            self.logger.error(f"Error retrieving features: {str(e)}")
            raise
    
    async def get_available_features(self) -> List[str]:
        """
        Get the list of available features.
        
        Returns:
            List of available feature names
        """
        try:
            # Send request to the Feature Store Service
            response = await self.client.send_request({
                "method": "GET",
                "path": "api/v1/features/available"
            })
            
            # Extract feature names
            if "data" in response and isinstance(response["data"], list):
                return response["data"]
            else:
                return []
        except Exception as e:
            self.logger.error(f"Error retrieving available features: {str(e)}")
            raise
    
    async def get_feature_metadata(
        self,
        feature_name: str
    ) -> Dict[str, Any]:
        """
        Get metadata about the specified feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Dictionary containing metadata about the feature
        """
        try:
            # Send request to the Feature Store Service
            response = await self.client.send_request({
                "method": "GET",
                "path": f"api/v1/features/metadata/{feature_name}"
            })
            
            # Extract metadata
            if "data" in response and isinstance(response["data"], dict):
                return response["data"]
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error retrieving feature metadata: {str(e)}")
            raise


class FeatureStoreAdapter(IFeatureStore):
    """
    Adapter implementation for the Feature Store interface.
    
    This adapter uses the HTTP service client to communicate with the Feature Store Service.
    """
    
    def __init__(
        self,
        config: ServiceClientConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Feature Store adapter.
        
        Args:
            config: Configuration for the service client
            logger: Logger to use (if None, creates a new logger)
        """
        self.client = AsyncHTTPServiceClient(config)
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def store_feature(
        self,
        feature_name: str,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
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
            # Prepare data for storage
            data_records = data.reset_index().to_dict(orient="records")
            
            # Prepare request body
            request_body = {
                "feature_name": feature_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "data": data_records
            }
            
            # Add metadata if provided
            if metadata:
                request_body["metadata"] = metadata
            
            # Send request to the Feature Store Service
            response = await self.client.send_request({
                "method": "POST",
                "path": "api/v1/features/store",
                "json": request_body
            })
            
            # Return success status
            return response.get("success", False)
        except Exception as e:
            self.logger.error(f"Error storing feature: {str(e)}")
            return False
    
    async def delete_feature(
        self,
        feature_name: str,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> bool:
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
            # Prepare request parameters
            params = {
                "feature_name": feature_name
            }
            if symbol:
                params["symbol"] = symbol
            if timeframe:
                params["timeframe"] = timeframe
            
            # Send request to the Feature Store Service
            response = await self.client.send_request({
                "method": "DELETE",
                "path": "api/v1/features",
                "params": params
            })
            
            # Return success status
            return response.get("success", False)
        except Exception as e:
            self.logger.error(f"Error deleting feature: {str(e)}")
            return False
    
    async def update_feature_metadata(
        self,
        feature_name: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for a feature.
        
        Args:
            feature_name: Name of the feature
            metadata: Updated metadata
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            # Send request to the Feature Store Service
            response = await self.client.send_request({
                "method": "PUT",
                "path": f"api/v1/features/metadata/{feature_name}",
                "json": {
                    "metadata": metadata
                }
            })
            
            # Return success status
            return response.get("success", False)
        except Exception as e:
            self.logger.error(f"Error updating feature metadata: {str(e)}")
            return False


class FeatureGeneratorAdapter(IFeatureGenerator):
    """
    Adapter implementation for the Feature Generator interface.
    
    This adapter uses the HTTP service client to communicate with the Feature Store Service.
    """
    
    def __init__(
        self,
        config: ServiceClientConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Feature Generator adapter.
        
        Args:
            config: Configuration for the service client
            logger: Logger to use (if None, creates a new logger)
        """
        self.client = AsyncHTTPServiceClient(config)
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def generate_feature(
        self,
        feature_name: str,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        parameters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
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
            # Prepare request body
            request_body = {
                "feature_name": feature_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            # Add parameters if provided
            if parameters:
                request_body["parameters"] = parameters
            
            # Send request to the Feature Store Service
            response = await self.client.send_request({
                "method": "POST",
                "path": "api/v1/features/generate",
                "json": request_body
            })
            
            # Convert response to DataFrame
            if "data" in response and response["data"]:
                df = pd.DataFrame(response["data"])
                
                # Convert timestamp to datetime
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)
                
                return df
            else:
                # Return empty DataFrame with feature name as column
                return pd.DataFrame(columns=[feature_name])
        except Exception as e:
            self.logger.error(f"Error generating feature: {str(e)}")
            raise
    
    async def get_available_generators(self) -> List[str]:
        """
        Get the list of available feature generators.
        
        Returns:
            List of available feature generator names
        """
        try:
            # Send request to the Feature Store Service
            response = await self.client.send_request({
                "method": "GET",
                "path": "api/v1/features/generators"
            })
            
            # Extract generator names
            if "data" in response and isinstance(response["data"], list):
                return response["data"]
            else:
                return []
        except Exception as e:
            self.logger.error(f"Error retrieving available generators: {str(e)}")
            raise
    
    async def get_generator_parameters(
        self,
        generator_name: str
    ) -> Dict[str, Any]:
        """
        Get the parameters for a feature generator.
        
        Args:
            generator_name: Name of the feature generator
            
        Returns:
            Dictionary containing the parameters for the generator
        """
        try:
            # Send request to the Feature Store Service
            response = await self.client.send_request({
                "method": "GET",
                "path": f"api/v1/features/generators/{generator_name}/parameters"
            })
            
            # Extract parameters
            if "data" in response and isinstance(response["data"], dict):
                return response["data"]
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error retrieving generator parameters: {str(e)}")
            raise