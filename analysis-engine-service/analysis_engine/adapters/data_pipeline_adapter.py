"""
Data Pipeline Adapter Module

This module provides adapters for data pipeline functionality,
helping to break circular dependencies between services.
"""
import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import httpx
import asyncio
import os

logger = logging.getLogger(__name__)


class TickDataServiceAdapter:
    """
    Adapter for TickDataService that provides standalone functionality
    to avoid circular dependencies.
    """
    
    def __init__(self, service_instance=None, config: Dict[str, Any] = None):
        """
        Initialize the adapter.
        
        Args:
            service_instance: Optional service instance to wrap
            config: Configuration parameters
        """
        self.service = service_instance
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Get data pipeline URL from config or environment
        data_pipeline_base_url = self.config.get(
            "data_pipeline_base_url", 
            os.environ.get("DATA_PIPELINE_BASE_URL", "http://data-pipeline-service:8000")
        )
        
        # Set up the client with resolved URL
        self.client = httpx.AsyncClient(
            base_url=f"{data_pipeline_base_url.rstrip('/')}/api/v1",
            timeout=30.0
        )
        
        # Data cache
        self.data_cache = {}
        self.last_update = {}
        self.cache_ttl = self.config.get("cache_ttl_minutes", 30)
    
    async def get_tick_data(self, 
                          symbol: str, 
                          start_time: datetime, 
                          end_time: datetime,
                          limit: int = 1000) -> pd.DataFrame:
        """
        Get tick data for a symbol within a time range.
        
        Args:
            symbol: Trading symbol
            start_time: Start time
            end_time: End time
            limit: Maximum number of ticks to return
            
        Returns:
            DataFrame with tick data
        """
        if self.service:
            try:
                return await self.service.get_tick_data(symbol, start_time, end_time, limit)
            except Exception as e:
                self.logger.warning(f"Error getting tick data from service: {str(e)}")
        
        # Check cache first
        cache_key = f"{symbol}_{start_time.isoformat()}_{end_time.isoformat()}_{limit}"
        if cache_key in self.data_cache:
            cache_age = datetime.now() - self.last_update.get(cache_key, datetime.min)
            if cache_age.total_seconds() < (self.cache_ttl * 60):
                return self.data_cache[cache_key]
        
        # Fallback to API call
        try:
            params = {
                "symbol": symbol,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "limit": limit
            }
            
            response = await self.client.get("/tick-data", params=params)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data.get("ticks", []))
            
            # Convert timestamp to datetime
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                
            # Cache the result
            self.data_cache[cache_key] = df
            self.last_update[cache_key] = datetime.now()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching tick data from API: {str(e)}")
            # Return empty DataFrame as fallback
            return pd.DataFrame()
    
    async def get_latest_ticks(self, symbol: str, count: int = 100) -> pd.DataFrame:
        """
        Get the latest ticks for a symbol.
        
        Args:
            symbol: Trading symbol
            count: Number of ticks to return
            
        Returns:
            DataFrame with latest ticks
        """
        if self.service:
            try:
                return await self.service.get_latest_ticks(symbol, count)
            except Exception as e:
                self.logger.warning(f"Error getting latest ticks from service: {str(e)}")
        
        # Fallback to API call
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)  # Look back 1 hour
        
        return await self.get_tick_data(symbol, start_time, end_time, count)
    
    async def get_tick_stream(self, symbol: str, callback: callable):
        """
        Get a stream of ticks for a symbol.
        
        Args:
            symbol: Trading symbol
            callback: Callback function to handle new ticks
        """
        if self.service:
            try:
                return await self.service.get_tick_stream(symbol, callback)
            except Exception as e:
                self.logger.warning(f"Error getting tick stream from service: {str(e)}")
        
        # Fallback implementation - simulate streaming with polling
        async def polling_stream():
            last_timestamp = datetime.now() - timedelta(minutes=1)
            
            while True:
                try:
                    # Get ticks since last timestamp
                    end_time = datetime.now()
                    ticks = await self.get_tick_data(symbol, last_timestamp, end_time, 100)
                    
                    if not ticks.empty:
                        # Update last timestamp
                        last_timestamp = ticks["timestamp"].max()
                        
                        # Call callback with new ticks
                        callback(ticks)
                    
                    # Wait before next poll
                    await asyncio.sleep(1.0)
                    
                except Exception as e:
                    self.logger.error(f"Error in tick polling stream: {str(e)}")
                    await asyncio.sleep(5.0)  # Wait longer after error
        
        # Start polling task
        task = asyncio.create_task(polling_stream())
        return task
    
    async def get_aggregated_ticks(self, 
                                 symbol: str, 
                                 start_time: datetime, 
                                 end_time: datetime,
                                 interval: str = "1m") -> pd.DataFrame:
        """
        Get aggregated tick data (OHLCV) for a symbol.
        
        Args:
            symbol: Trading symbol
            start_time: Start time
            end_time: End time
            interval: Aggregation interval
            
        Returns:
            DataFrame with aggregated tick data
        """
        if self.service:
            try:
                return await self.service.get_aggregated_ticks(symbol, start_time, end_time, interval)
            except Exception as e:
                self.logger.warning(f"Error getting aggregated ticks from service: {str(e)}")
        
        # Fallback to API call
        try:
            params = {
                "symbol": symbol,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "interval": interval
            }
            
            response = await self.client.get("/ohlcv", params=params)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data.get("candles", []))
            
            # Convert timestamp to datetime
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching aggregated tick data from API: {str(e)}")
            # Return empty DataFrame as fallback
            return pd.DataFrame()
