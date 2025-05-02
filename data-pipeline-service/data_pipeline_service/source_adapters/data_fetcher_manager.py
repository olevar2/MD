"""
DataFetcherManager module.

Coordinates data source adapters and provides a unified interface for fetching data.
"""
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type, Union

from common_lib.exceptions import (DataFetchError, DataTransformationError,  # Updated import
                                   DataValidationError, ServiceError)
from common_lib.schemas import OHLCVData, SourceConfig, TickData
from common_lib.services.config_manager import ConfigManager
from core_foundations.utils.logger import get_logger
from data_pipeline_service.models.schemas import OHLCVData, TickData
from data_pipeline_service.source_adapters.base_adapter import (
    DataSourceAdapter, OHLCVDataSourceAdapter, TickDataSourceAdapter
)

# Initialize logger
logger = get_logger("data-fetcher-manager")


class DataFetcherManager:
    """
    Coordinates data source adapters and provides a unified interface for fetching data.
    
    This manager handles connections to multiple data sources, authentication, rate limiting,
    and provides a simple interface for the rest of the system to retrieve data.
    """
    
    def __init__(self):
        """Initialize the data fetcher manager."""
        self._adapters: Dict[str, DataSourceAdapter] = {}
        self._default_ohlcv_adapter: Optional[str] = None
        self._default_tick_adapter: Optional[str] = None
    
    def register_adapter(
        self, adapter_id: str, adapter: DataSourceAdapter, 
        set_as_default: bool = False
    ) -> None:
        """
        Register a data source adapter.
        
        Args:
            adapter_id: Unique identifier for the adapter
            adapter: Data source adapter instance
            set_as_default: If True, set as default adapter for its type
        """
        self._adapters[adapter_id] = adapter
        
        if set_as_default:
            if isinstance(adapter, OHLCVDataSourceAdapter):
                self._default_ohlcv_adapter = adapter_id
                logger.info(f"Set {adapter_id} as default OHLCV adapter")
                
            if isinstance(adapter, TickDataSourceAdapter):
                self._default_tick_adapter = adapter_id
                logger.info(f"Set {adapter_id} as default tick data adapter")
        
        logger.info(f"Registered adapter: {adapter_id}")
    
    def unregister_adapter(self, adapter_id: str) -> None:
        """
        Unregister a data source adapter.
        
        Args:
            adapter_id: Unique identifier for the adapter
        """
        if adapter_id in self._adapters:
            del self._adapters[adapter_id]
            
            if self._default_ohlcv_adapter == adapter_id:
                self._default_ohlcv_adapter = None
                
            if self._default_tick_adapter == adapter_id:
                self._default_tick_adapter = None
                
            logger.info(f"Unregistered adapter: {adapter_id}")
    
    def get_adapter(self, adapter_id: str) -> Optional[DataSourceAdapter]:
        """
        Get a registered adapter by ID.
        
        Args:
            adapter_id: Unique identifier for the adapter
            
        Returns:
            The adapter instance, or None if not found
        """
        return self._adapters.get(adapter_id)
    
    def get_default_ohlcv_adapter(self) -> Optional[OHLCVDataSourceAdapter]:
        """
        Get the default OHLCV data adapter.
        
        Returns:
            The default OHLCV adapter, or None if not set
        """
        if not self._default_ohlcv_adapter:
            return None
            
        adapter = self._adapters.get(self._default_ohlcv_adapter)
        if not adapter or not isinstance(adapter, OHLCVDataSourceAdapter):
            return None
            
        return adapter
    
    def get_default_tick_adapter(self) -> Optional[TickDataSourceAdapter]:
        """
        Get the default tick data adapter.
        
        Returns:
            The default tick data adapter, or None if not set
        """
        if not self._default_tick_adapter:
            return None
            
        adapter = self._adapters.get(self._default_tick_adapter)
        if not adapter or not isinstance(adapter, TickDataSourceAdapter):
            return None
            
        return adapter
    
    async def connect_all(self) -> Dict[str, bool]:
        """
        Connect to all registered data sources.
        
        Returns:
            Dictionary mapping adapter IDs to connection status (True for success, False for failure)
        """
        results: Dict[str, bool] = {}
        
        for adapter_id, adapter in self._adapters.items():
            try:
                results[adapter_id] = await adapter.connect()
                if results[adapter_id]:
                    logger.info(f"Connected to {adapter_id}")
                else:
                    logger.warning(f"Failed to connect to {adapter_id}")
            except Exception as e:
                logger.error(f"Error connecting to {adapter_id}: {str(e)}")
                results[adapter_id] = False
        
        return results
    
    async def disconnect_all(self) -> None:
        """Disconnect from all registered data sources."""
        for adapter_id, adapter in self._adapters.items():
            try:
                await adapter.disconnect()
                logger.info(f"Disconnected from {adapter_id}")
            except Exception as e:
                logger.error(f"Error disconnecting from {adapter_id}: {str(e)}")
    
    async def get_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        from_time: datetime,
        to_time: datetime,
        limit: Optional[int] = None,
        adapter_id: Optional[str] = None,
    ) -> List[OHLCVData]:
        """
        Get OHLCV data from a data source.
        
        Args:
            symbol: Trading instrument symbol
            timeframe: Candle timeframe
            from_time: Start time for data query
            to_time: End time for data query
            limit: Maximum number of candles to return
            adapter_id: Specific adapter to use, or None for default
            
        Returns:
            List of OHLCV data objects
        """
        # Ensure timezone awareness
        if from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=timezone.utc)
        if to_time.tzinfo is None:
            to_time = to_time.replace(tzinfo=timezone.utc)
        
        # Get the appropriate adapter
        adapter = None
        if adapter_id:
            adapter = self.get_adapter(adapter_id)
            if not adapter or not isinstance(adapter, OHLCVDataSourceAdapter):
                raise ValueError(f"No valid OHLCV adapter found with ID: {adapter_id}")
        else:
            adapter = self.get_default_ohlcv_adapter()
            if not adapter:
                raise ValueError("No default OHLCV adapter set")
        
        # Ensure the adapter is connected
        if not await adapter.is_connected():
            await adapter.connect()
        
        # Fetch the data
        try:
            data_dicts = await adapter.get_ohlcv_data(
                symbol, timeframe, from_time, to_time, limit
            )
            
            # Convert to OHLCVData objects
            return [
                OHLCVData(
                    symbol=d["symbol"],
                    timestamp=d["timestamp"],
                    timeframe=d["timeframe"],
                    open=d["open"],
                    high=d["high"],
                    low=d["low"],
                    close=d["close"],
                    volume=d["volume"]
                )
                for d in data_dicts
            ]
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {str(e)}")
            raise DataFetchError(
                f"Failed to fetch OHLCV data for {symbol}",
                source=adapter_id or "default"
            ) from e
    
    async def get_tick_data(
        self,
        symbol: str,
        from_time: datetime,
        to_time: datetime,
        limit: Optional[int] = None,
        adapter_id: Optional[str] = None,
    ) -> List[TickData]:
        """
        Get tick data from a data source.
        
        Args:
            symbol: Trading instrument symbol
            from_time: Start time for data query
            to_time: End time for data query
            limit: Maximum number of ticks to return
            adapter_id: Specific adapter to use, or None for default
            
        Returns:
            List of tick data objects
        """
        # Ensure timezone awareness
        if from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=timezone.utc)
        if to_time.tzinfo is None:
            to_time = to_time.replace(tzinfo=timezone.utc)
        
        # Get the appropriate adapter
        adapter = None
        if adapter_id:
            adapter = self.get_adapter(adapter_id)
            if not adapter or not isinstance(adapter, TickDataSourceAdapter):
                raise ValueError(f"No valid tick data adapter found with ID: {adapter_id}")
        else:
            adapter = self.get_default_tick_adapter()
            if not adapter:
                raise ValueError("No default tick data adapter set")
        
        # Ensure the adapter is connected
        if not await adapter.is_connected():
            await adapter.connect()
        
        # Fetch the data
        try:
            # Check if the adapter has a direct method to get TickData objects
            if hasattr(adapter, 'get_tick_data_objects'):
                return await adapter.get_tick_data_objects(
                    symbol, from_time, to_time, limit
                )
            
            # Otherwise convert from dictionaries
            data_dicts = await adapter.get_tick_data(
                symbol, from_time, to_time, limit
            )
            
            return [
                TickData(
                    symbol=d["symbol"],
                    timestamp=d["timestamp"],
                    bid=d["bid"],
                    ask=d["ask"],
                    bid_volume=d.get("bid_volume", 0.0),
                    ask_volume=d.get("ask_volume", 0.0)
                )
                for d in data_dicts
            ]
        except Exception as e:
            logger.error(f"Error fetching tick data: {str(e)}")
            raise DataFetchError(
                f"Failed to fetch tick data for {symbol}",
                source=adapter_id or "default"
            ) from e
    
    async def get_available_instruments(
        self, adapter_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get available instruments from a data source.
        
        Args:
            adapter_id: Specific adapter to use, or None for default
            
        Returns:
            List of instrument dictionaries
        """
        adapter = None
        
        if adapter_id:
            adapter = self.get_adapter(adapter_id)
            if not adapter:
                raise ValueError(f"No adapter found with ID: {adapter_id}")
        else:
            # Try to use default OHLCV adapter, then default tick adapter
            adapter = self.get_default_ohlcv_adapter() or self.get_default_tick_adapter()
            if not adapter:
                raise ValueError("No default adapters set")
        
        # Ensure the adapter is connected
        if not await adapter.is_connected():
            await adapter.connect()
        
        return await adapter.get_instruments()
    
    async def health_check(self) -> Dict[str, Dict[str, Any]]:
        """
        Check the health of all registered adapters.
        
        Returns:
            Dictionary mapping adapter IDs to health status
        """
        results = {}
        
        for adapter_id, adapter in self._adapters.items():
            try:
                is_connected = await adapter.is_connected()
                results[adapter_id] = {
                    "status": "connected" if is_connected else "disconnected",
                    "type": adapter.__class__.__name__,
                    "is_default_ohlcv": adapter_id == self._default_ohlcv_adapter,
                    "is_default_tick": adapter_id == self._default_tick_adapter
                }
            except Exception as e:
                results[adapter_id] = {
                    "status": "error",
                    "message": str(e),
                    "type": adapter.__class__.__name__
                }
        
        return results
    
    async def subscribe_to_ticks(
        self, symbol: str, adapter_id: Optional[str] = None
    ) -> bool:
        """
        Subscribe to real-time tick data for a specific instrument.
        
        Args:
            symbol: Trading instrument symbol
            adapter_id: Specific adapter to use, or None for default
            
        Returns:
            True if subscription was successful, False otherwise
        """
        adapter = None
        
        if adapter_id:
            adapter = self.get_adapter(adapter_id)
            if not adapter or not isinstance(adapter, TickDataSourceAdapter):
                raise ValueError(f"No valid tick data adapter found with ID: {adapter_id}")
        else:
            adapter = self.get_default_tick_adapter()
            if not adapter:
                raise ValueError("No default tick data adapter set")
        
        # Ensure the adapter is connected
        if not await adapter.is_connected():
            await adapter.connect()
        
        return await adapter.subscribe_to_ticks(symbol)
    
    async def unsubscribe_from_ticks(
        self, symbol: str, adapter_id: Optional[str] = None
    ) -> bool:
        """
        Unsubscribe from real-time tick data for a specific instrument.
        
        Args:
            symbol: Trading instrument symbol
            adapter_id: Specific adapter to use, or None for default
            
        Returns:
            True if unsubscription was successful, False otherwise
        """
        adapter = None
        
        if adapter_id:
            adapter = self.get_adapter(adapter_id)
            if not adapter or not isinstance(adapter, TickDataSourceAdapter):
                raise ValueError(f"No valid tick data adapter found with ID: {adapter_id}")
        else:
            adapter = self.get_default_tick_adapter()
            if not adapter:
                raise ValueError("No default tick data adapter set")
        
        return await adapter.unsubscribe_from_ticks(symbol)

    async def fetch_historical_ohlcv(self, symbol: str, start_date: datetime, end_date: datetime, timeframe: str) -> List[OHLCVData]:
        """
        Fetch historical OHLCV data for a symbol from configured sources.

        Args:
            symbol: Trading instrument symbol
            start_date: Start date for data query
            end_date: End date for data query
            timeframe: Candle timeframe

        Returns:
            List of OHLCV data objects

        Raises:
            DataFetchError: If data fetching or validation fails for all sources
        """
        results = {}
        errors = {}
        adapters = self._get_adapters_for_symbol(symbol, data_type='ohlcv')

        if not adapters:
            raise ConfigurationError(f"No adapters configured or available for OHLCV data for symbol {symbol}")

        tasks = []
        adapter_names = list(adapters.keys()) # Store names for associating results

        for source_name, adapter in adapters.items():
             tasks.append(self._fetch_historical_data_async(adapter, source_name, symbol, start_date, end_date, timeframe))

        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(task_results):
            source_name = adapter_names[i]
            if isinstance(result, Exception):
                # Log specific errors caught within _fetch_historical_data_async or wrap unexpected ones
                if isinstance(result, (DataFetchError, DataValidationError, ServiceError)):
                     self.logger.error(f"Error fetching OHLCV data from {source_name} for {symbol}: {result}")
                     errors[source_name] = result
                else:
                    self.logger.exception(f"Unexpected exception fetching OHLCV data from {source_name} for {symbol}: {result}")
                    errors[source_name] = DataFetchError(f"Unexpected error from {source_name}: {result}", source=source_name)
            elif result: # Ensure result is not None or empty
                results[source_name] = result
                self.logger.info(f"Successfully fetched and validated {len(result)} OHLCV records for {symbol} from {source_name}")
            else:
                 self.logger.warning(f"No OHLCV data returned for {symbol} from {source_name} between {start_date} and {end_date}")


        if not results and errors:
            first_error = next(iter(errors.values()))
            raise DataFetchError(f"Failed to fetch data for {symbol} from all sources. First error: {first_error}", details=errors) from first_error
        elif not results:
             raise DataFetchError(f"No data could be fetched for {symbol} from any configured source.")


        # --- Data Processing/Merging Logic ---
        try:
            processed_data = self._process_fetched_data(results, symbol, timeframe)
            return processed_data
        except DataTransformationError as e: # Catch specific transformation errors
            self.logger.error(f"Error processing fetched data for {symbol}: {e}")
            raise # Re-raise
        except Exception as e: # Catch unexpected processing errors
            self.logger.exception(f"Unexpected error processing fetched data for {symbol}: {e}")
            raise DataTransformationError(f"Unexpected error processing data for {symbol}: {e}", transformation="DataFetcherManager Aggregation") from e


    async def fetch_tick_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[TickData]:
        """
        Fetch historical tick data for a symbol from configured sources.

        Args:
            symbol: Trading instrument symbol
            start_date: Start date for data query
            end_date: End date for data query

        Returns:
            List of Tick data objects

        Raises:
            DataFetchError: If data fetching or validation fails for all sources
        """
        results = {}
        errors = {}
        adapters = self._get_adapters_for_symbol(symbol, data_type='tick')

        if not adapters:
            raise ConfigurationError(f"No adapters configured or available for tick data for symbol {symbol}")

        tasks = []
        adapter_names = list(adapters.keys()) # Store names for associating results

        for source_name, adapter in adapters.items():
             tasks.append(self._fetch_tick_data_async(adapter, source_name, symbol, start_date, end_date))

        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(task_results):
            source_name = adapter_names[i]
            if isinstance(result, Exception):
                # Log specific errors caught within _fetch_tick_data_async or wrap unexpected ones
                if isinstance(result, (DataFetchError, DataValidationError, ServiceError)):
                     self.logger.error(f"Error fetching tick data from {source_name} for {symbol}: {result}")
                     errors[source_name] = result
                else:
                    self.logger.exception(f"Unexpected exception fetching tick data from {source_name} for {symbol}: {result}")
                    errors[source_name] = DataFetchError(f"Unexpected error from {source_name}: {result}", source=source_name)
            elif result: # Ensure result is not None or empty
                results[source_name] = result
                self.logger.info(f"Successfully fetched and validated {len(result)} tick records for {symbol} from {source_name}")
            else:
                 self.logger.warning(f"No tick data returned for {symbol} from {source_name} between {start_date} and {end_date}")


        if not results and errors:
            first_error = next(iter(errors.values()))
            raise DataFetchError(f"Failed to fetch tick data for {symbol} from all sources. First error: {first_error}", details=errors) from first_error
        elif not results:
             raise DataFetchError(f"No tick data could be fetched for {symbol} from any configured source.")

        # --- Data Processing/Merging Logic ---
        try:
            processed_data = self._process_fetched_data(results, symbol, 'tick') # Assuming 'tick' timeframe identifier
            return processed_data
        except DataTransformationError as e:
            self.logger.error(f"Error processing fetched tick data for {symbol}: {e}")
            raise
        except Exception as e:
            self.logger.exception(f"Unexpected error processing fetched tick data for {symbol}: {e}")
            raise DataTransformationError(f"Unexpected error processing tick data for {symbol}: {e}", transformation="DataFetcherManager Tick Aggregation") from e


    async def _fetch_historical_data_async(self, adapter: BaseAdapter, source_name: str, symbol: str, start_date: datetime, end_date: datetime, timeframe: str) -> List[OHLCVData]:
        """Helper coroutine to fetch and validate historical data from a single adapter."""
        try:
            raw_data = await adapter.get_historical_data(symbol, start_date, end_date, timeframe)
            if raw_data is None:
                return []
            validated_data = self.validation_engine.validate_ohlcv(raw_data, symbol, timeframe, source_name)
            return validated_data
        except (DataFetchError, DataValidationError, ServiceError) as e:
             # Logged by the caller, just raise to propagate
             raise e
        except Exception as e:
            # Log and wrap unexpected errors from this specific adapter call
            self.logger.exception(f"Unexpected error in adapter {source_name} for {symbol} OHLCV: {e}")
            raise DataFetchError(f"Unexpected adapter error: {e}", source=source_name) from e


    async def _fetch_tick_data_async(self, adapter: BaseAdapter, source_name: str, symbol: str, start_date: datetime, end_date: datetime) -> List[TickData]:
        """Helper coroutine to fetch and validate tick data from a