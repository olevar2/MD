"""
Market Data Adapter Module

This module implements the adapter pattern for the market data service,
using the interfaces defined in common-lib.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
import asyncio

from common_lib.interfaces.market_data import IMarketDataProvider, IMarketDataCache
from common_lib.errors.base_exceptions import (
    BaseError, ErrorCode, ValidationError, DataError, ServiceError
)

from data_pipeline_service.services.market_data_service import MarketDataService
from data_pipeline_service.repositories.market_data_repository import MarketDataRepository
from data_pipeline_service.caching.market_data_cache import MarketDataCache
from data_pipeline_service.config.settings import get_settings

# Configure logging
logger = logging.getLogger(__name__)


class MarketDataProviderAdapter(IMarketDataProvider):
    """
    Adapter for the MarketDataService to implement the IMarketDataProvider interface.
    
    This adapter allows the MarketDataService to be used through the standardized
    IMarketDataProvider interface, enabling better service integration and
    reducing circular dependencies.
    """
    
    def __init__(self, market_data_service: Optional[MarketDataService] = None):
        """
        Initialize the MarketDataProviderAdapter.
        
        Args:
            market_data_service: Optional MarketDataService instance. If not provided,
                                a new instance will be created.
        """
        self._market_data_service = market_data_service or MarketDataService(
            repository=MarketDataRepository(),
            cache=MarketDataCache()
        )
        self._settings = get_settings()
        logger.info("MarketDataProviderAdapter initialized")
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get historical market data for a symbol.
        
        Args:
            symbol: The trading symbol (e.g., "EURUSD")
            timeframe: The timeframe (e.g., "1m", "5m", "1h", "1d")
            start_time: Start time for the data
            end_time: Optional end time for the data
            
        Returns:
            DataFrame containing the historical data
            
        Raises:
            ValidationError: If the input parameters are invalid
            DataError: If there's an issue with the data
            ServiceError: If there's a service-related error
        """
        try:
            # Validate inputs
            if not symbol:
                raise ValidationError("Symbol cannot be empty", field="symbol")
            
            if not timeframe:
                raise ValidationError("Timeframe cannot be empty", field="timeframe")
            
            if not start_time:
                raise ValidationError("Start time cannot be empty", field="start_time")
            
            # Call the service method
            result = await self._market_data_service.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )
            
            return result
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Convert other exceptions to appropriate error types
            if "not found" in str(e).lower():
                raise DataError(
                    f"No data found for {symbol} with timeframe {timeframe}",
                    error_code=ErrorCode.DATA_MISSING_ERROR,
                    data_source="market_data",
                    data_type="historical",
                    cause=e
                )
            elif "database" in str(e).lower():
                raise ServiceError(
                    f"Database error while fetching historical data for {symbol}",
                    error_code=ErrorCode.SERVICE_DEPENDENCY_ERROR,
                    service_name="market_data_service",
                    operation="get_historical_data",
                    cause=e
                )
            else:
                raise ServiceError(
                    f"Error fetching historical data for {symbol}: {str(e)}",
                    error_code=ErrorCode.SERVICE_UNAVAILABLE,
                    service_name="market_data_service",
                    operation="get_historical_data",
                    cause=e
                )
    
    async def get_latest_price(self, symbol: str) -> Dict[str, float]:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: The trading symbol (e.g., "EURUSD")
            
        Returns:
            Dictionary containing the latest price information
            
        Raises:
            ValidationError: If the input parameters are invalid
            DataError: If there's an issue with the data
            ServiceError: If there's a service-related error
        """
        try:
            # Validate inputs
            if not symbol:
                raise ValidationError("Symbol cannot be empty", field="symbol")
            
            # Call the service method
            result = await self._market_data_service.get_latest_price(symbol=symbol)
            
            return result
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Convert other exceptions to appropriate error types
            if "not found" in str(e).lower():
                raise DataError(
                    f"No latest price found for {symbol}",
                    error_code=ErrorCode.DATA_MISSING_ERROR,
                    data_source="market_data",
                    data_type="latest_price",
                    cause=e
                )
            else:
                raise ServiceError(
                    f"Error fetching latest price for {symbol}: {str(e)}",
                    error_code=ErrorCode.SERVICE_UNAVAILABLE,
                    service_name="market_data_service",
                    operation="get_latest_price",
                    cause=e
                )
    
    async def get_symbols(self) -> List[str]:
        """
        Get available symbols.
        
        Returns:
            List of available symbols
            
        Raises:
            ServiceError: If there's a service-related error
        """
        try:
            # Call the service method
            result = await self._market_data_service.get_symbols()
            
            return result
        except Exception as e:
            # Convert exceptions to appropriate error types
            raise ServiceError(
                f"Error fetching available symbols: {str(e)}",
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                service_name="market_data_service",
                operation="get_symbols",
                cause=e
            )


class MarketDataCacheAdapter(IMarketDataCache):
    """
    Adapter for the MarketDataCache to implement the IMarketDataCache interface.
    
    This adapter allows the MarketDataCache to be used through the standardized
    IMarketDataCache interface, enabling better service integration and
    reducing circular dependencies.
    """
    
    def __init__(self, market_data_cache: Optional[MarketDataCache] = None):
        """
        Initialize the MarketDataCacheAdapter.
        
        Args:
            market_data_cache: Optional MarketDataCache instance. If not provided,
                              a new instance will be created.
        """
        self._market_data_cache = market_data_cache or MarketDataCache()
        self._settings = get_settings()
        logger.info("MarketDataCacheAdapter initialized")
    
    async def get_cached_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get cached market data if available.
        
        Args:
            symbol: The trading symbol (e.g., "EURUSD")
            timeframe: The timeframe (e.g., "1m", "5m", "1h", "1d")
            start_time: Start time for the data
            end_time: Optional end time for the data
            
        Returns:
            DataFrame containing the cached data, or None if not in cache
            
        Raises:
            ValidationError: If the input parameters are invalid
        """
        try:
            # Validate inputs
            if not symbol:
                raise ValidationError("Symbol cannot be empty", field="symbol")
            
            if not timeframe:
                raise ValidationError("Timeframe cannot be empty", field="timeframe")
            
            if not start_time:
                raise ValidationError("Start time cannot be empty", field="start_time")
            
            # Call the cache method
            result = await self._market_data_cache.get(
                key=f"{symbol}_{timeframe}_{start_time.isoformat()}_{end_time.isoformat() if end_time else 'none'}"
            )
            
            return result
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Log the error but return None (cache miss)
            logger.warning(f"Error accessing cache for {symbol} {timeframe}: {str(e)}")
            return None
    
    async def cache_data(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame
    ) -> bool:
        """
        Cache market data.
        
        Args:
            symbol: The trading symbol (e.g., "EURUSD")
            timeframe: The timeframe (e.g., "1m", "5m", "1h", "1d")
            data: The data to cache
            
        Returns:
            True if caching was successful, False otherwise
            
        Raises:
            ValidationError: If the input parameters are invalid
        """
        try:
            # Validate inputs
            if not symbol:
                raise ValidationError("Symbol cannot be empty", field="symbol")
            
            if not timeframe:
                raise ValidationError("Timeframe cannot be empty", field="timeframe")
            
            if data is None or data.empty:
                raise ValidationError("Data cannot be empty", field="data")
            
            # Extract time range from data
            start_time = data.index.min()
            end_time = data.index.max()
            
            # Call the cache method
            key = f"{symbol}_{timeframe}_{start_time.isoformat()}_{end_time.isoformat()}"
            await self._market_data_cache.set(key=key, value=data)
            
            return True
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Log the error and return False (cache operation failed)
            logger.error(f"Error caching data for {symbol} {timeframe}: {str(e)}")
            return False
    
    async def invalidate_cache(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> bool:
        """
        Invalidate cache entries.
        
        Args:
            symbol: Optional symbol to invalidate cache for
            timeframe: Optional timeframe to invalidate cache for
            
        Returns:
            True if invalidation was successful, False otherwise
        """
        try:
            # Call the cache method
            if symbol and timeframe:
                # Invalidate specific symbol and timeframe
                pattern = f"{symbol}_{timeframe}_*"
                await self._market_data_cache.invalidate_pattern(pattern=pattern)
            elif symbol:
                # Invalidate all timeframes for a symbol
                pattern = f"{symbol}_*"
                await self._market_data_cache.invalidate_pattern(pattern=pattern)
            elif timeframe:
                # Invalidate all symbols for a timeframe
                pattern = f"*_{timeframe}_*"
                await self._market_data_cache.invalidate_pattern(pattern=pattern)
            else:
                # Invalidate all market data cache
                await self._market_data_cache.invalidate_all()
            
            return True
        except Exception as e:
            # Log the error and return False (invalidation failed)
            logger.error(f"Error invalidating cache: {str(e)}")
            return False
