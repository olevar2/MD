"""
Analysis Engine Client for Strategy Execution Engine

This module provides a client for interacting with the Analysis Engine Service.
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

import aiohttp
from fastapi import HTTPException, status

from strategy_execution_engine.core.config import get_settings
from strategy_execution_engine.error import (
    ServiceError,
    DataFetchError,
    async_with_error_handling
)

logger = logging.getLogger(__name__)

class AnalysisEngineClient:
    """
    Client for interacting with the Analysis Engine Service.
    """
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the Analysis Engine Service
            api_key: API key for authentication
        """
        settings = get_settings()
        self.base_url = base_url or settings.analysis_engine_url
        self.api_key = api_key or settings.analysis_engine_key
        self.session = None
        self.logger = logger
        
        self.logger.info(f"Initialized Analysis Engine Client with base URL: {self.base_url}")
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """
        Ensure that a session exists.
        
        Returns:
            aiohttp.ClientSession: Client session
        """
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": self.api_key
                }
            )
        
        return self.session
    
    async def close(self) -> None:
        """
        Close the client session.
        """
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
    @async_with_error_handling
    async def get_technical_analysis(self, instrument: str, timeframe: str, 
                                    indicators: List[str], start_date: str, 
                                    end_date: str) -> Dict[str, Any]:
        """
        Get technical analysis for an instrument.
        
        Args:
            instrument: Instrument symbol
            timeframe: Timeframe
            indicators: List of indicators to calculate
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dict: Technical analysis results
            
        Raises:
            DataFetchError: If data fetch fails
            ServiceError: If service returns an error
        """
        session = await self._ensure_session()
        
        url = f"{self.base_url}/api/v1/analysis/technical"
        
        params = {
            "instrument": instrument,
            "timeframe": timeframe,
            "indicators": ",".join(indicators),
            "start_date": start_date,
            "end_date": end_date
        }
        
        try:
            self.logger.debug(f"Fetching technical analysis from {url} with params: {params}")
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_text = await response.text()
                    self.logger.error(f"Error fetching technical analysis: {error_text}")
                    
                    if response.status == 404:
                        raise DataFetchError(f"Data not found for {instrument} ({timeframe})")
                    else:
                        raise ServiceError(f"Analysis Engine Service error: {error_text}")
        except aiohttp.ClientError as e:
            self.logger.error(f"Error connecting to Analysis Engine Service: {e}")
            raise ServiceError(f"Error connecting to Analysis Engine Service: {str(e)}")
    
    @async_with_error_handling
    async def get_market_regime(self, instrument: str, timeframe: str) -> Dict[str, Any]:
        """
        Get market regime for an instrument.
        
        Args:
            instrument: Instrument symbol
            timeframe: Timeframe
            
        Returns:
            Dict: Market regime information
            
        Raises:
            DataFetchError: If data fetch fails
            ServiceError: If service returns an error
        """
        session = await self._ensure_session()
        
        url = f"{self.base_url}/api/v1/market-regime"
        
        params = {
            "instrument": instrument,
            "timeframe": timeframe
        }
        
        try:
            self.logger.debug(f"Fetching market regime from {url} with params: {params}")
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_text = await response.text()
                    self.logger.error(f"Error fetching market regime: {error_text}")
                    
                    if response.status == 404:
                        raise DataFetchError(f"Data not found for {instrument} ({timeframe})")
                    else:
                        raise ServiceError(f"Analysis Engine Service error: {error_text}")
        except aiohttp.ClientError as e:
            self.logger.error(f"Error connecting to Analysis Engine Service: {e}")
            raise ServiceError(f"Error connecting to Analysis Engine Service: {str(e)}")
    
    @async_with_error_handling
    async def get_pattern_recognition(self, instrument: str, timeframe: str, 
                                     patterns: List[str], start_date: str, 
                                     end_date: str) -> Dict[str, Any]:
        """
        Get pattern recognition for an instrument.
        
        Args:
            instrument: Instrument symbol
            timeframe: Timeframe
            patterns: List of patterns to recognize
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dict: Pattern recognition results
            
        Raises:
            DataFetchError: If data fetch fails
            ServiceError: If service returns an error
        """
        session = await self._ensure_session()
        
        url = f"{self.base_url}/api/v1/analysis/patterns"
        
        params = {
            "instrument": instrument,
            "timeframe": timeframe,
            "patterns": ",".join(patterns),
            "start_date": start_date,
            "end_date": end_date
        }
        
        try:
            self.logger.debug(f"Fetching pattern recognition from {url} with params: {params}")
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error_text = await response.text()
                    self.logger.error(f"Error fetching pattern recognition: {error_text}")
                    
                    if response.status == 404:
                        raise DataFetchError(f"Data not found for {instrument} ({timeframe})")
                    else:
                        raise ServiceError(f"Analysis Engine Service error: {error_text}")
        except aiohttp.ClientError as e:
            self.logger.error(f"Error connecting to Analysis Engine Service: {e}")
            raise ServiceError(f"Error connecting to Analysis Engine Service: {str(e)}")
    
    @async_with_error_handling
    async def check_health(self) -> Dict[str, Any]:
        """
        Check the health of the Analysis Engine Service.
        
        Returns:
            Dict: Health check result
            
        Raises:
            ServiceError: If service is unhealthy or unreachable
        """
        session = await self._ensure_session()
        
        url = f"{self.base_url}/health"
        
        try:
            self.logger.debug(f"Checking health of Analysis Engine Service at {url}")
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "healthy",
                        "message": "Connection successful",
                        "details": data
                    }
                else:
                    error_text = await response.text()
                    self.logger.error(f"Analysis Engine Service health check failed: {error_text}")
                    return {
                        "status": "unhealthy",
                        "message": f"Health check failed with status {response.status}",
                        "details": error_text
                    }
        except aiohttp.ClientError as e:
            self.logger.error(f"Error connecting to Analysis Engine Service: {e}")
            return {
                "status": "unhealthy",
                "message": f"Connection failed: {str(e)}",
                "details": None
            }
