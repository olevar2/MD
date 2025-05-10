"""
Base Alternative Data Adapter.

This module provides the base implementation for alternative data adapters.
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from common_lib.exceptions import DataFetchError, DataValidationError
from common_lib.interfaces.alternative_data import IAlternativeDataProvider
from data_management_service.alternative.models import (
    AlternativeDataType,
    DataFrequency,
    DataReliability,
    DataSourceMetadata
)

logger = logging.getLogger(__name__)


class BaseAlternativeDataAdapter(IAlternativeDataProvider, ABC):
    """Base class for alternative data adapters."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the adapter.

        Args:
            config: Configuration for the adapter
        """
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.source_id = config.get("source_id", "default")
        self.metadata = self._build_metadata(config)
        self.supported_data_types = self._get_supported_data_types()
        
        # Initialize cache
        self.cache = {}
        self.cache_ttl = config.get("cache_ttl", 3600)  # Default 1 hour
        
        logger.info(f"Initialized {self.name} adapter with source_id={self.source_id}")

    def _build_metadata(self, config: Dict[str, Any]) -> DataSourceMetadata:
        """
        Build metadata for the data source.

        Args:
            config: Configuration for the adapter

        Returns:
            DataSourceMetadata object
        """
        return DataSourceMetadata(
            name=config.get("name", self.__class__.__name__),
            description=config.get("description", ""),
            provider=config.get("provider", "unknown"),
            frequency=DataFrequency(config.get("frequency", "daily")),
            reliability=DataReliability(config.get("reliability", "medium")),
            last_updated=datetime.utcnow(),
            coverage_start=config.get("coverage_start"),
            coverage_end=config.get("coverage_end"),
            attributes=config.get("attributes", {})
        )

    @abstractmethod
    def _get_supported_data_types(self) -> List[str]:
        """
        Get the list of data types supported by this adapter.

        Returns:
            List of supported data types
        """
        pass

    async def get_available_data_types(self) -> List[str]:
        """
        Get the list of available alternative data types from this provider.

        Returns:
            List of available data types
        """
        return self.supported_data_types

    async def get_metadata(self, data_type: str) -> Dict[str, Any]:
        """
        Get metadata about the specified data type.

        Args:
            data_type: Type of alternative data

        Returns:
            Dictionary containing metadata about the data type
        """
        if data_type not in self.supported_data_types:
            raise ValueError(f"Data type '{data_type}' is not supported by this adapter")
        
        # Return base metadata plus any data type specific metadata
        return {
            **self.metadata.dict(),
            "data_type": data_type,
            "source_id": self.source_id
        }

    def _get_cache_key(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> str:
        """
        Generate a cache key for the given parameters.

        Args:
            data_type: Type of alternative data
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters

        Returns:
            Cache key string
        """
        symbols_str = ",".join(sorted(symbols))
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()
        
        # Sort kwargs by key for consistent cache keys
        kwargs_str = "&".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        
        return f"{data_type}|{symbols_str}|{start_str}|{end_str}|{kwargs_str}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if the cached data for the given key is still valid.

        Args:
            cache_key: Cache key

        Returns:
            True if cache is valid, False otherwise
        """
        if cache_key not in self.cache:
            return False
        
        cache_entry = self.cache[cache_key]
        cache_time = cache_entry.get("timestamp")
        
        if not cache_time:
            return False
        
        # Check if cache has expired
        now = datetime.utcnow()
        age = (now - cache_time).total_seconds()
        
        return age < self.cache_ttl

    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Get data from cache if available and valid.

        Args:
            cache_key: Cache key

        Returns:
            Cached data or None if not available
        """
        if not self._is_cache_valid(cache_key):
            return None
        
        return self.cache[cache_key].get("data")

    def _store_in_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        """
        Store data in cache.

        Args:
            cache_key: Cache key
            data: Data to cache
        """
        self.cache[cache_key] = {
            "data": data,
            "timestamp": datetime.utcnow()
        }

    @abstractmethod
    async def _fetch_data(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch data from the source.

        Args:
            data_type: Type of alternative data
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters

        Returns:
            DataFrame containing the data
        """
        pass

    async def get_data(
        self,
        data_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Retrieve alternative data for the specified parameters.

        Args:
            data_type: Type of alternative data to retrieve
            symbols: List of symbols to retrieve data for
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            **kwargs: Additional parameters specific to the data type

        Returns:
            DataFrame containing the alternative data
        """
        if data_type not in self.supported_data_types:
            raise ValueError(f"Data type '{data_type}' is not supported by this adapter")
        
        # Check cache first
        use_cache = kwargs.pop("use_cache", True)
        if use_cache:
            cache_key = self._get_cache_key(data_type, symbols, start_date, end_date, **kwargs)
            cached_data = self._get_from_cache(cache_key)
            
            if cached_data is not None:
                logger.debug(f"Retrieved {data_type} data from cache for {symbols}")
                return cached_data
        
        try:
            # Fetch data from source
            data = await self._fetch_data(data_type, symbols, start_date, end_date, **kwargs)
            
            # Store in cache if caching is enabled
            if use_cache:
                self._store_in_cache(cache_key, data)
            
            return data
        except Exception as e:
            logger.error(f"Error fetching {data_type} data for {symbols}: {str(e)}")
            raise DataFetchError(f"Failed to fetch {data_type} data: {str(e)}")
