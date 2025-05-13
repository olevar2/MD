"""
Analysis Engine Service Adapter.

This module provides adapter implementations for the Analysis Engine Service interfaces.
These adapters allow other services to interact with the Analysis Engine Service
without direct dependencies, breaking circular dependencies.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from common_lib.interfaces.analysis_engine import IAnalysisProvider, IIndicatorProvider, IPatternRecognizer
from common_lib.service_client.base_client import ServiceClientConfig
from common_lib.service_client.http_client import AsyncHTTPServiceClient


class AnalysisProviderAdapter(IAnalysisProvider):
    """
    Adapter implementation for the Analysis Provider interface.
    
    This adapter uses the HTTP service client to communicate with the Analysis Engine Service.
    """
    
    def __init__(
        self,
        config: ServiceClientConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Analysis Provider adapter.
        
        Args:
            config: Configuration for the service client
            logger: Logger to use (if None, creates a new logger)
        """
        self.client = AsyncHTTPServiceClient(config)
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def analyze_market(
        self,
        symbol: str,
        timeframe: str,
        analysis_type: str,
        start_time: datetime,
        end_time: datetime,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform market analysis for the specified parameters.
        
        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            timeframe: Timeframe for the data (e.g., "1m", "5m", "1h", "1d")
            analysis_type: Type of analysis to perform
            start_time: Start time for analysis
            end_time: End time for analysis
            parameters: Optional parameters for the analysis
            
        Returns:
            Dictionary containing the analysis results
        """
        try:
            # Prepare request body
            request_body = {
                "symbol": symbol,
                "timeframe": timeframe,
                "analysis_type": analysis_type,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            # Add parameters if provided
            if parameters:
                request_body["parameters"] = parameters
            
            # Send request to the Analysis Engine Service
            response = await self.client.send_request({
                "method": "POST",
                "path": "api/v1/analysis",
                "json": request_body
            })
            
            # Extract analysis results
            if "data" in response and isinstance(response["data"], dict):
                return response["data"]
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error performing market analysis: {str(e)}")
            raise
    
    async def get_available_analysis_types(self) -> List[str]:
        """
        Get the list of available analysis types.
        
        Returns:
            List of available analysis types
        """
        try:
            # Send request to the Analysis Engine Service
            response = await self.client.send_request({
                "method": "GET",
                "path": "api/v1/analysis/types"
            })
            
            # Extract analysis types
            if "data" in response and isinstance(response["data"], list):
                return response["data"]
            else:
                return []
        except Exception as e:
            self.logger.error(f"Error retrieving available analysis types: {str(e)}")
            raise
    
    async def get_analysis_parameters(
        self,
        analysis_type: str
    ) -> Dict[str, Any]:
        """
        Get the parameters for an analysis type.
        
        Args:
            analysis_type: Type of analysis
            
        Returns:
            Dictionary containing the parameters for the analysis
        """
        try:
            # Send request to the Analysis Engine Service
            response = await self.client.send_request({
                "method": "GET",
                "path": f"api/v1/analysis/parameters/{analysis_type}"
            })
            
            # Extract parameters
            if "data" in response and isinstance(response["data"], dict):
                return response["data"]
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error retrieving analysis parameters: {str(e)}")
            raise


class IndicatorProviderAdapter(IIndicatorProvider):
    """
    Adapter implementation for the Indicator Provider interface.
    
    This adapter uses the HTTP service client to communicate with the Analysis Engine Service.
    """
    
    def __init__(
        self,
        config: ServiceClientConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Indicator Provider adapter.
        
        Args:
            config: Configuration for the service client
            logger: Logger to use (if None, creates a new logger)
        """
        self.client = AsyncHTTPServiceClient(config)
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def calculate_indicator(
        self,
        indicator_name: str,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        parameters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Calculate an indicator for the specified parameters.
        
        Args:
            indicator_name: Name of the indicator to calculate
            symbol: Trading symbol (e.g., "EUR/USD")
            timeframe: Timeframe for the data (e.g., "1m", "5m", "1h", "1d")
            start_time: Start time for calculation
            end_time: End time for calculation
            parameters: Optional parameters for the indicator
            
        Returns:
            DataFrame containing the indicator values
        """
        try:
            # Prepare request body
            request_body = {
                "indicator_name": indicator_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            # Add parameters if provided
            if parameters:
                request_body["parameters"] = parameters
            
            # Send request to the Analysis Engine Service
            response = await self.client.send_request({
                "method": "POST",
                "path": "api/v1/indicators/calculate",
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
                # Return empty DataFrame
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error calculating indicator: {str(e)}")
            raise
    
    async def calculate_indicators(
        self,
        indicator_configs: List[Dict[str, Any]],
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Calculate multiple indicators for the specified parameters.
        
        Args:
            indicator_configs: List of indicator configurations
            symbol: Trading symbol (e.g., "EUR/USD")
            timeframe: Timeframe for the data (e.g., "1m", "5m", "1h", "1d")
            start_time: Start time for calculation
            end_time: End time for calculation
            
        Returns:
            DataFrame containing the indicator values
        """
        try:
            # Prepare request body
            request_body = {
                "indicator_configs": indicator_configs,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            # Send request to the Analysis Engine Service
            response = await self.client.send_request({
                "method": "POST",
                "path": "api/v1/indicators/calculate-batch",
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
                # Return empty DataFrame
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            raise
    
    async def get_available_indicators(self) -> List[str]:
        """
        Get the list of available indicators.
        
        Returns:
            List of available indicator names
        """
        try:
            # Send request to the Analysis Engine Service
            response = await self.client.send_request({
                "method": "GET",
                "path": "api/v1/indicators/available"
            })
            
            # Extract indicator names
            if "data" in response and isinstance(response["data"], list):
                return response["data"]
            else:
                return []
        except Exception as e:
            self.logger.error(f"Error retrieving available indicators: {str(e)}")
            raise
    
    async def get_indicator_parameters(
        self,
        indicator_name: str
    ) -> Dict[str, Any]:
        """
        Get the parameters for an indicator.
        
        Args:
            indicator_name: Name of the indicator
            
        Returns:
            Dictionary containing the parameters for the indicator
        """
        try:
            # Send request to the Analysis Engine Service
            response = await self.client.send_request({
                "method": "GET",
                "path": f"api/v1/indicators/parameters/{indicator_name}"
            })
            
            # Extract parameters
            if "data" in response and isinstance(response["data"], dict):
                return response["data"]
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error retrieving indicator parameters: {str(e)}")
            raise


class PatternRecognizerAdapter(IPatternRecognizer):
    """
    Adapter implementation for the Pattern Recognizer interface.
    
    This adapter uses the HTTP service client to communicate with the Analysis Engine Service.
    """
    
    def __init__(
        self,
        config: ServiceClientConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Pattern Recognizer adapter.
        
        Args:
            config: Configuration for the service client
            logger: Logger to use (if None, creates a new logger)
        """
        self.client = AsyncHTTPServiceClient(config)
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def recognize_patterns(
        self,
        symbol: str,
        timeframe: str,
        pattern_types: List[str],
        start_time: datetime,
        end_time: datetime,
        parameters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Recognize patterns in market data.
        
        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            timeframe: Timeframe for the data (e.g., "1m", "5m", "1h", "1d")
            pattern_types: List of pattern types to recognize
            start_time: Start time for pattern recognition
            end_time: End time for pattern recognition
            parameters: Optional parameters for pattern recognition
            
        Returns:
            DataFrame containing the recognized patterns
        """
        try:
            # Prepare request body
            request_body = {
                "symbol": symbol,
                "timeframe": timeframe,
                "pattern_types": pattern_types,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            # Add parameters if provided
            if parameters:
                request_body["parameters"] = parameters
            
            # Send request to the Analysis Engine Service
            response = await self.client.send_request({
                "method": "POST",
                "path": "api/v1/patterns/recognize",
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
                # Return empty DataFrame
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error recognizing patterns: {str(e)}")
            raise
    
    async def get_available_pattern_types(self) -> List[str]:
        """
        Get the list of available pattern types.
        
        Returns:
            List of available pattern types
        """
        try:
            # Send request to the Analysis Engine Service
            response = await self.client.send_request({
                "method": "GET",
                "path": "api/v1/patterns/types"
            })
            
            # Extract pattern types
            if "data" in response and isinstance(response["data"], list):
                return response["data"]
            else:
                return []
        except Exception as e:
            self.logger.error(f"Error retrieving available pattern types: {str(e)}")
            raise