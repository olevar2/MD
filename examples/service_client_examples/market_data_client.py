"""
Market Data Client Example

This module demonstrates how to implement a service client using the standardized template.
"""

import logging
from typing import Dict, Any, Optional, Union, List
from datetime import datetime

from common_lib.clients.base_client import BaseServiceClient, ClientConfig
from common_lib.clients.templates.service_client_template import StandardServiceClient
from common_lib.clients.exceptions import (
    ClientError,
    ClientConnectionError,
    ClientTimeoutError,
    ClientValidationError,
    ClientAuthenticationError
)


class MarketDataClient(StandardServiceClient):
    """
    Client for interacting with the Market Data Service.
    
    This client provides methods for:
    1. Retrieving market data (OHLCV, ticks)
    2. Getting instrument information
    3. Subscribing to market data updates
    """
    
    def __init__(self, config: Union[ClientConfig, Dict[str, Any]]):
        """
        Initialize the market data client.
        
        Args:
            config: Client configuration
        """
        super().__init__(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def get_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get OHLCV (Open, High, Low, Close, Volume) data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'EUR/USD')
            timeframe: Timeframe for the data (e.g., '1m', '1h', '1d')
            start_time: Start time for the data
            end_time: End time for the data
            limit: Maximum number of data points to return
            
        Returns:
            OHLCV data
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientValidationError: If the request is invalid
            ClientAuthenticationError: If authentication fails
        """
        self.logger.debug(f"Getting OHLCV data for {symbol} ({timeframe})")
        
        # Prepare parameters
        params = {
            "symbol": symbol,
            "timeframe": timeframe
        }
        
        if start_time:
            params["start_time"] = start_time.isoformat()
        
        if end_time:
            params["end_time"] = end_time.isoformat()
        
        if limit:
            params["limit"] = limit
        
        try:
            return await self.get("market-data/ohlcv", params=params)
        except Exception as e:
            self.logger.error(f"Failed to get OHLCV data for {symbol}: {str(e)}")
            # Re-raise the exception to be handled by the caller
            raise
    
    async def get_tick_data(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get tick data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'EUR/USD')
            start_time: Start time for the data
            end_time: End time for the data
            limit: Maximum number of data points to return
            
        Returns:
            Tick data
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientValidationError: If the request is invalid
            ClientAuthenticationError: If authentication fails
        """
        self.logger.debug(f"Getting tick data for {symbol}")
        
        # Prepare parameters
        params = {
            "symbol": symbol
        }
        
        if start_time:
            params["start_time"] = start_time.isoformat()
        
        if end_time:
            params["end_time"] = end_time.isoformat()
        
        if limit:
            params["limit"] = limit
        
        try:
            return await self.get("market-data/ticks", params=params)
        except Exception as e:
            self.logger.error(f"Failed to get tick data for {symbol}: {str(e)}")
            # Re-raise the exception to be handled by the caller
            raise
    
    async def get_instrument(self, symbol: str) -> Dict[str, Any]:
        """
        Get information about a trading instrument.
        
        Args:
            symbol: Trading symbol (e.g., 'EUR/USD')
            
        Returns:
            Instrument information
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientValidationError: If the request is invalid
            ClientAuthenticationError: If authentication fails
        """
        self.logger.debug(f"Getting instrument information for {symbol}")
        
        try:
            return await self.get(f"market-data/instruments/{symbol}")
        except Exception as e:
            self.logger.error(f"Failed to get instrument information for {symbol}: {str(e)}")
            # Re-raise the exception to be handled by the caller
            raise
    
    async def list_instruments(
        self,
        category: Optional[str] = None,
        active_only: bool = True
    ) -> Dict[str, Any]:
        """
        List available trading instruments.
        
        Args:
            category: Filter by instrument category (e.g., 'forex', 'crypto')
            active_only: Whether to return only active instruments
            
        Returns:
            List of instruments
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientValidationError: If the request is invalid
            ClientAuthenticationError: If authentication fails
        """
        self.logger.debug(f"Listing instruments (category={category}, active_only={active_only})")
        
        # Prepare parameters
        params = {
            "active_only": active_only
        }
        
        if category:
            params["category"] = category
        
        try:
            return await self.get("market-data/instruments", params=params)
        except Exception as e:
            self.logger.error(f"Failed to list instruments: {str(e)}")
            # Re-raise the exception to be handled by the caller
            raise
    
    async def get_latest_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'EUR/USD')
            
        Returns:
            Latest price information
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientValidationError: If the request is invalid
            ClientAuthenticationError: If authentication fails
        """
        self.logger.debug(f"Getting latest price for {symbol}")
        
        try:
            return await self.get(f"market-data/prices/{symbol}/latest")
        except Exception as e:
            self.logger.error(f"Failed to get latest price for {symbol}: {str(e)}")
            # Re-raise the exception to be handled by the caller
            raise