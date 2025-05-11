"""
Market Data Service Adapter.

This module provides adapter implementations for the Market Data Service interfaces.
These adapters allow other services to interact with the Market Data Service
without direct dependencies, breaking circular dependencies.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable

import pandas as pd

from common_lib.interfaces.market_data import IMarketDataProvider, IMarketDataCache
from common_lib.service_client.base_client import ServiceClientConfig
from common_lib.service_client.http_client import AsyncHTTPServiceClient


class MarketDataProviderAdapter(IMarketDataProvider):
    """
    Adapter implementation for the Market Data Provider interface.
    
    This adapter uses the HTTP service client to communicate with the Market Data Service.
    """
    
    def __init__(
        self,
        config: ServiceClientConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Market Data Provider adapter.
        
        Args:
            config: Configuration for the service client
            logger: Logger to use (if None, creates a new logger)
        """
        self.client = AsyncHTTPServiceClient(config)
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._subscriptions = {}
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        include_incomplete_candles: bool = False
    ) -> pd.DataFrame:
        """
        Retrieve historical market data for the specified parameters.
        
        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            timeframe: Timeframe for the data (e.g., "1m", "5m", "1h", "1d")
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            include_incomplete_candles: Whether to include incomplete candles
            
        Returns:
            DataFrame containing the historical market data with OHLCV columns
        """
        try:
            # Prepare request parameters
            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "include_incomplete_candles": include_incomplete_candles
            }
            
            # Send request to the Market Data Service
            response = await self.client.send_request({
                "method": "GET",
                "path": "api/v1/historical-data",
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
                # Return empty DataFrame with expected columns
                return pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume"]
                )
        except Exception as e:
            self.logger.error(f"Error retrieving historical data: {str(e)}")
            raise
    
    async def get_latest_price(
        self,
        symbol: str
    ) -> Dict[str, float]:
        """
        Get the latest price for the specified symbol.
        
        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            
        Returns:
            Dictionary containing bid and ask prices
        """
        try:
            # Send request to the Market Data Service
            response = await self.client.send_request({
                "method": "GET",
                "path": f"api/v1/latest-price/{symbol}"
            })
            
            # Extract price data
            if "data" in response:
                return {
                    "bid": response["data"].get("bid", 0.0),
                    "ask": response["data"].get("ask", 0.0)
                }
            else:
                return {"bid": 0.0, "ask": 0.0}
        except Exception as e:
            self.logger.error(f"Error retrieving latest price: {str(e)}")
            raise
    
    async def get_available_symbols(self) -> List[str]:
        """
        Get the list of available trading symbols.
        
        Returns:
            List of available symbols
        """
        try:
            # Send request to the Market Data Service
            response = await self.client.send_request({
                "method": "GET",
                "path": "api/v1/symbols"
            })
            
            # Extract symbols
            if "data" in response and isinstance(response["data"], list):
                return response["data"]
            else:
                return []
        except Exception as e:
            self.logger.error(f"Error retrieving available symbols: {str(e)}")
            raise
    
    async def get_available_timeframes(self) -> List[str]:
        """
        Get the list of available timeframes.
        
        Returns:
            List of available timeframes
        """
        try:
            # Send request to the Market Data Service
            response = await self.client.send_request({
                "method": "GET",
                "path": "api/v1/timeframes"
            })
            
            # Extract timeframes
            if "data" in response and isinstance(response["data"], list):
                return response["data"]
            else:
                return []
        except Exception as e:
            self.logger.error(f"Error retrieving available timeframes: {str(e)}")
            raise
    
    async def subscribe_to_price_updates(
        self,
        symbol: str,
        callback: Callable
    ) -> str:
        """
        Subscribe to real-time price updates for the specified symbol.
        
        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            callback: Callback function to be called when a price update is received
            
        Returns:
            Subscription ID
        """
        try:
            # Send subscription request to the Market Data Service
            response = await self.client.send_request({
                "method": "POST",
                "path": "api/v1/subscribe",
                "json": {
                    "symbol": symbol,
                    "type": "price"
                }
            })
            
            # Extract subscription ID
            subscription_id = response.get("subscription_id", "")
            
            # Store callback
            if subscription_id:
                self._subscriptions[subscription_id] = callback
            
            return subscription_id
        except Exception as e:
            self.logger.error(f"Error subscribing to price updates: {str(e)}")
            raise
    
    async def unsubscribe_from_price_updates(
        self,
        subscription_id: str
    ) -> bool:
        """
        Unsubscribe from real-time price updates.
        
        Args:
            subscription_id: Subscription ID returned by subscribe_to_price_updates
            
        Returns:
            True if unsubscribed successfully, False otherwise
        """
        try:
            # Send unsubscription request to the Market Data Service
            response = await self.client.send_request({
                "method": "POST",
                "path": "api/v1/unsubscribe",
                "json": {
                    "subscription_id": subscription_id
                }
            })
            
            # Remove callback
            if subscription_id in self._subscriptions:
                del self._subscriptions[subscription_id]
            
            # Return success status
            return response.get("success", False)
        except Exception as e:
            self.logger.error(f"Error unsubscribing from price updates: {str(e)}")
            return False


class MarketDataCacheAdapter(IMarketDataCache):
    """
    Adapter implementation for the Market Data Cache interface.
    
    This adapter uses the HTTP service client to communicate with the Market Data Service's cache.
    """
    
    def __init__(
        self,
        config: ServiceClientConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Market Data Cache adapter.
        
        Args:
            config: Configuration for the service client
            logger: Logger to use (if None, creates a new logger)
        """
        self.client = AsyncHTTPServiceClient(config)
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def get_cached_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve cached market data for the specified parameters.
        
        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            timeframe: Timeframe for the data (e.g., "1m", "5m", "1h", "1d")
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            
        Returns:
            DataFrame containing the cached market data, or None if not cached
        """
        try:
            # Prepare request parameters
            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            # Send request to the Market Data Service
            response = await self.client.send_request({
                "method": "GET",
                "path": "api/v1/cache/data",
                "params": params
            })
            
            # Check if data is cached
            if response.get("cached", False) and "data" in response:
                # Convert response to DataFrame
                df = pd.DataFrame(response["data"])
                
                # Convert timestamp to datetime
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)
                
                return df
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error retrieving cached data: {str(e)}")
            return None
    
    async def cache_data(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame
    ) -> bool:
        """
        Cache market data for the specified parameters.
        
        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            timeframe: Timeframe for the data (e.g., "1m", "5m", "1h", "1d")
            data: DataFrame containing the market data to cache
            
        Returns:
            True if cached successfully, False otherwise
        """
        try:
            # Prepare data for caching
            data_records = data.reset_index().to_dict(orient="records")
            
            # Send request to the Market Data Service
            response = await self.client.send_request({
                "method": "POST",
                "path": "api/v1/cache/data",
                "json": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "data": data_records
                }
            })
            
            # Return success status
            return response.get("success", False)
        except Exception as e:
            self.logger.error(f"Error caching data: {str(e)}")
            return False
    
    async def invalidate_cache(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> bool:
        """
        Invalidate cached market data.
        
        Args:
            symbol: Trading symbol to invalidate, or None to invalidate all symbols
            timeframe: Timeframe to invalidate, or None to invalidate all timeframes
            
        Returns:
            True if invalidated successfully, False otherwise
        """
        try:
            # Prepare request parameters
            params = {}
            if symbol:
                params["symbol"] = symbol
            if timeframe:
                params["timeframe"] = timeframe
            
            # Send request to the Market Data Service
            response = await self.client.send_request({
                "method": "POST",
                "path": "api/v1/cache/invalidate",
                "json": params
            })
            
            # Return success status
            return response.get("success", False)
        except Exception as e:
            self.logger.error(f"Error invalidating cache: {str(e)}")
            return False