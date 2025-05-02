"""
Analysis Engine Client

This module provides a client for interacting with the Analysis Engine Service.
It uses the standardized client implementation from common-lib.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from datetime import datetime

from common_lib.clients import BaseServiceClient, ClientConfig
from common_lib.clients.exceptions import ClientError

logger = logging.getLogger(__name__)


class AnalysisEngineClient(BaseServiceClient):
    """
    Client for interacting with the Analysis Engine Service.
    
    This client provides methods for:
    1. Retrieving technical indicators
    2. Detecting market regimes
    3. Analyzing market conditions
    4. Accessing adaptive layer functionality
    """
    
    def __init__(self, config: Union[ClientConfig, Dict[str, Any]]):
        """
        Initialize the Analysis Engine client.
        
        Args:
            config: Client configuration
        """
        super().__init__(config)
        logger.info(f"Analysis Engine Client initialized with base URL: {self.base_url}")
    
    async def get_technical_indicators(
        self,
        symbol: str,
        timeframe: str,
        indicators: List[Dict[str, Any]],
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Get technical indicators for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            indicators: List of indicator configurations
            start_time: Start time for data
            end_time: End time for data
            
        Returns:
            Dictionary containing indicator data
            
        Raises:
            ClientError: If the request fails
        """
        # Format datetime objects to ISO strings if provided
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()
        
        # Prepare request data
        data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "indicators": indicators
        }
        
        if start_time:
            data["start_time"] = start_time
        if end_time:
            data["end_time"] = end_time
        
        try:
            response = await self.post("indicators/calculate", data=data)
            return response
        except Exception as e:
            logger.error(f"Error getting technical indicators: {str(e)}")
            raise ClientError(
                f"Failed to get technical indicators for {symbol} {timeframe}",
                service_name=self.config.service_name
            ) from e
    
    async def detect_market_regime(
        self,
        symbol: str,
        timeframe: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect the current market regime.
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            data: Optional price data (if not provided, will be fetched from the service)
            
        Returns:
            Dictionary containing regime information
            
        Raises:
            ClientError: If the request fails
        """
        # Prepare request data
        request_data = {
            "symbol": symbol,
            "timeframe": timeframe
        }
        
        if data:
            request_data["data"] = data
        
        try:
            response = await self.post("market/regime", data=request_data)
            return response
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            raise ClientError(
                f"Failed to detect market regime for {symbol} {timeframe}",
                service_name=self.config.service_name
            ) from e
    
    async def analyze_market_conditions(
        self,
        symbol: str,
        timeframe: str,
        analysis_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze current market conditions.
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            analysis_types: Types of analysis to perform
            
        Returns:
            Dictionary containing analysis results
            
        Raises:
            ClientError: If the request fails
        """
        # Default analysis types if not provided
        if not analysis_types:
            analysis_types = ["trend", "volatility", "support_resistance"]
        
        # Prepare request data
        data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis_types": analysis_types
        }
        
        try:
            response = await self.post("market/analyze", data=data)
            return response
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {str(e)}")
            raise ClientError(
                f"Failed to analyze market conditions for {symbol} {timeframe}",
                service_name=self.config.service_name
            ) from e
    
    async def get_multi_timeframe_analysis(
        self,
        symbol: str,
        timeframes: List[str],
        analysis_types: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get analysis across multiple timeframes.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes to analyze
            analysis_types: Types of analysis to perform
            
        Returns:
            Dictionary containing analysis results for each timeframe
            
        Raises:
            ClientError: If the request fails
        """
        # Default analysis types if not provided
        if not analysis_types:
            analysis_types = ["trend", "volatility", "support_resistance"]
        
        # Prepare request data
        data = {
            "symbol": symbol,
            "timeframes": timeframes,
            "analysis_types": analysis_types
        }
        
        try:
            response = await self.post("market/multi-timeframe", data=data)
            return response
        except Exception as e:
            logger.error(f"Error getting multi-timeframe analysis: {str(e)}")
            raise ClientError(
                f"Failed to get multi-timeframe analysis for {symbol}",
                service_name=self.config.service_name
            ) from e
