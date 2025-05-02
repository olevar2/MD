"""
Base interface for data source adapters.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from core_foundations.interfaces.base_interfaces import IService


class DataSourceAdapter(IService, ABC):
    """
    Base interface for all data source adapters.
    
    A data source adapter is responsible for connecting to a specific data source,
    retrieving data, and converting it to a standard format for the data pipeline.
    """
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the data source.
        
        Returns:
            True if connection was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """
        Check if the adapter is connected to the data source.
        
        Returns:
            True if connected, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_instruments(self) -> List[Dict[str, Any]]:
        """
        Get list of available instruments from the data source.
        
        Returns:
            List of instruments with their properties
        """
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information from the data source.
        
        Returns:
            Account information dictionary
        """
        pass


class OHLCVDataSourceAdapter(DataSourceAdapter, ABC):
    """
    Interface for OHLCV data source adapters.
    """
    
    @abstractmethod
    async def get_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        from_time: datetime,
        to_time: datetime,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve OHLCV data for a specific instrument and timeframe.
        
        Args:
            symbol: Trading instrument symbol
            timeframe: Candle timeframe
            from_time: Start time for data query
            to_time: End time for data query
            limit: Maximum number of candles to return (optional)
            
        Returns:
            List of OHLCV data dictionaries
        """
        pass


class TickDataSourceAdapter(DataSourceAdapter, ABC):
    """
    Interface for tick data source adapters.
    """
    
    @abstractmethod
    async def get_tick_data(
        self,
        symbol: str,
        from_time: datetime,
        to_time: datetime,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve tick data for a specific instrument.
        
        Args:
            symbol: Trading instrument symbol
            from_time: Start time for data query
            to_time: End time for data query
            limit: Maximum number of ticks to return (optional)
            
        Returns:
            List of tick data dictionaries
        """
        pass
    
    @abstractmethod
    async def subscribe_to_ticks(self, symbol: str) -> bool:
        """
        Subscribe to real-time tick data for a specific instrument.
        
        Args:
            symbol: Trading instrument symbol
            
        Returns:
            True if subscription was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def unsubscribe_from_ticks(self, symbol: str) -> bool:
        """
        Unsubscribe from real-time tick data for a specific instrument.
        
        Args:
            symbol: Trading instrument symbol
            
        Returns:
            True if unsubscription was successful, False otherwise
        """
        pass