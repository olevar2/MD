"""
Dukascopy data source adapter implementation.

This adapter connects to Dukascopy's historical tick data service to retrieve high-quality tick data
for Forex and other instruments.
"""
import asyncio
import gzip
import io
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import httpx
import numpy as np
import pandas as pd
from common_lib.exceptions import DataFetchError, DataTransformationError
from common_lib.schemas import OHLCVData, TickData
from common_lib.utils.date_utils import convert_to_utc_datetime, ensure_timezone
from core_foundations.utils.logger import get_logger
from adapters.base_adapter import TickDataSourceAdapter
logger = get_logger('dukascopy-adapter')


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class DukascopyAdapter(TickDataSourceAdapter):
    """
    Adapter for Dukascopy historical tick data.
    
    This adapter provides access to Dukascopy's historical tick data archives,
    which offer high-quality, precise tick data for Forex pairs and other instruments.
    """
    BASE_URL = 'https://datafeed.dukascopy.com/datafeed'
    INSTRUMENT_MAPPING = {'EUR/USD': 'EURUSD', 'GBP/USD': 'GBPUSD',
        'USD/JPY': 'USDJPY', 'AUD/USD': 'AUDUSD', 'USD/CHF': 'USDCHF',
        'USD/CAD': 'USDCAD', 'NZD/USD': 'NZDUSD', 'EUR/GBP': 'EURGBP',
        'EUR/JPY': 'EURJPY', 'GBP/JPY': 'GBPJPY'}
    REVERSE_INSTRUMENT_MAPPING = {v: k for k, v in INSTRUMENT_MAPPING.items()}

    def __init__(self, session_timeout: int=30, max_retries: int=3,
        retry_delay: float=1.0):
        """
        Initialize the Dukascopy adapter.
        
        Args:
            session_timeout: HTTP session timeout in seconds
            max_retries: Maximum number of retry attempts for HTTP requests
            retry_delay: Delay between retry attempts in seconds
        """
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_timeout = session_timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._connected = False
        self._subscribed_symbols: Set[str] = set()

    async def connect(self) ->bool:
        """
        Initialize HTTP session for connecting to Dukascopy.
        
        Returns:
            True if connection setup was successful
        """
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.
                ClientTimeout(total=self._session_timeout))
        self._connected = True
        logger.info('Dukascopy adapter connected')
        return True

    async def disconnect(self) ->None:
        """Close HTTP session."""
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._connected = False
        self._subscribed_symbols.clear()
        logger.info('Dukascopy adapter disconnected')

    async def is_connected(self) ->bool:
        """
        Check if adapter is connected.
        
        For HTTP-based adapters, this mainly checks if the session exists.
        
        Returns:
            True if connected
        """
        return self._connected and self._session is not None

    async def get_instruments(self) ->List[Dict[str, Any]]:
        """
        Get list of available instruments from Dukascopy.
        
        Returns:
            List of instrument dictionaries
        """
        instruments = []
        for standard_symbol, dukascopy_symbol in self.INSTRUMENT_MAPPING.items(
            ):
            instruments.append({'symbol': standard_symbol, 'name':
                standard_symbol, 'type': 'forex', 'source_symbol':
                dukascopy_symbol, 'pip_size': 0.0001 if 'JPY' not in
                standard_symbol else 0.01})
        return instruments

    async def get_account_info(self) ->Dict[str, Any]:
        """
        Get account information.
        
        Since this is a data provider without an account concept,
        this method returns basic information about the data source.
        
        Returns:
            Data source information
        """
        return {'name': 'Dukascopy Historical Data', 'type':
            'historical_data_provider', 'instruments_count': len(self.
            INSTRUMENT_MAPPING)}

    @async_with_exception_handling
    async def _download_tick_data_for_hour(self, symbol: str, dt: datetime
        ) ->Optional[bytes]:
        """
        Download compressed tick data for a specific hour.
        
        Args:
            symbol: Instrument symbol in Dukascopy format
            dt: Hour timestamp
            
        Returns:
            Raw compressed data or None if not available
        """
        if not self._connected or self._session is None:
            raise DataFetchError('Not connected to Dukascopy')
        url = (
            f'{self.BASE_URL}/{symbol}/{dt.year}/{dt.month - 1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5'
            )
        for attempt in range(self._max_retries + 1):
            try:
                async with self._session.get(url) as response:
                    if response.status == 200:
                        return await response.read()
                    elif response.status == 404:
                        logger.debug(
                            f'No tick data available for {symbol} at {dt} (404 Not Found)'
                            )
                        return None
                    else:
                        logger.warning(
                            f'HTTP error {response.status} for {url}')
                        if attempt < self._max_retries:
                            await asyncio.sleep(self._retry_delay)
                            continue
                        return None
            except Exception as e:
                logger.error(f'Error downloading tick data: {str(e)}')
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_delay)
                    continue
                raise DataFetchError(
                    f'Failed to download tick data for {symbol} at {dt}',
                    source='Dukascopy') from e
        return None

    @async_with_exception_handling
    async def _decompress_and_parse_ticks(self, data: bytes, symbol: str,
        hour_dt: datetime) ->List[Dict[str, Any]]:
        """
        Decompress and parse binary tick data.
        
        Args:
            data: Compressed binary tick data
            symbol: Instrument symbol in Dukascopy format
            hour_dt: Base hour timestamp
            
        Returns:
            List of parsed tick data dictionaries
        """
        if not data:
            return []
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data), mode='rb') as f:
                decompressed_data = f.read()
            num_ticks = len(decompressed_data) // 20
            if num_ticks == 0:
                return []
            dt_type = np.dtype([('time', 'i4'), ('ask', 'i4'), ('bid', 'i4'
                ), ('ask_vol', 'f4'), ('bid_vol', 'f4')])
            tick_array = np.frombuffer(decompressed_data, dtype=dt_type)
            df = pd.DataFrame(tick_array)
            df['timestamp'] = df['time'].apply(lambda ms: hour_dt +
                timedelta(milliseconds=ms))
            divisor = 1000 if 'JPY' in self.REVERSE_INSTRUMENT_MAPPING.get(
                symbol, '') else 100000
            df['ask'] = df['ask'] / divisor
            df['bid'] = df['bid'] / divisor
            standard_symbol = self.REVERSE_INSTRUMENT_MAPPING.get(symbol,
                symbol)
            ticks = []
            for _, row in df.iterrows():
                ticks.append({'symbol': standard_symbol, 'timestamp': row[
                    'timestamp'].to_pydatetime().replace(tzinfo=timezone.
                    utc), 'bid': row['bid'], 'ask': row['ask'],
                    'bid_volume': row['bid_vol'], 'ask_volume': row['ask_vol']}
                    )
            return ticks
        except Exception as e:
            logger.error(f'Error parsing tick data: {str(e)}')
            raise DataFetchError(f'Failed to parse tick data for {symbol}',
                source='Dukascopy') from e

    async def get_tick_data(self, symbol: str, from_time: datetime, to_time:
        datetime, limit: Optional[int]=None) ->List[Dict[str, Any]]:
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
        if from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=timezone.utc)
        if to_time.tzinfo is None:
            to_time = to_time.replace(tzinfo=timezone.utc)
        dukascopy_symbol = self.INSTRUMENT_MAPPING.get(symbol)
        if not dukascopy_symbol:
            raise ValueError(f'Unsupported symbol: {symbol}')
        current_hour = datetime(from_time.year, from_time.month, from_time.
            day, from_time.hour, tzinfo=timezone.utc)
        end_hour = datetime(to_time.year, to_time.month, to_time.day,
            to_time.hour, tzinfo=timezone.utc)
        all_ticks = []
        while current_hour <= end_hour:
            raw_data = await self._download_tick_data_for_hour(dukascopy_symbol
                , current_hour)
            if raw_data:
                hour_ticks = await self._decompress_and_parse_ticks(raw_data,
                    dukascopy_symbol, current_hour)
                filtered_ticks = [tick for tick in hour_ticks if from_time <=
                    tick['timestamp'] <= to_time]
                all_ticks.extend(filtered_ticks)
                if limit and len(all_ticks) >= limit:
                    all_ticks = all_ticks[:limit]
                    break
            current_hour += timedelta(hours=1)
        logger.info(
            f'Retrieved {len(all_ticks)} ticks for {symbol} from Dukascopy')
        return all_ticks

    async def subscribe_to_ticks(self, symbol: str) ->bool:
        """
        Subscribe to real-time tick data for a specific instrument.
        
        Note: Dukascopy historical adapter doesn't support real-time data.
        This method is implemented for interface compatibility.
        
        Args:
            symbol: Trading instrument symbol
            
        Returns:
            Always False as real-time subscription is not supported
        """
        logger.warning(
            f'Real-time tick subscription not supported for {symbol}')
        return False

    async def unsubscribe_from_ticks(self, symbol: str) ->bool:
        """
        Unsubscribe from real-time tick data for a specific instrument.
        
        Note: Dukascopy historical adapter doesn't support real-time data.
        This method is implemented for interface compatibility.
        
        Args:
            symbol: Trading instrument symbol
            
        Returns:
            Always True as there's nothing to unsubscribe from
        """
        return True

    @classmethod
    def map_to_standard_symbol(cls, dukascopy_symbol: str) ->str:
        """
        Convert Dukascopy symbol to standard format.
        
        Args:
            dukascopy_symbol: Symbol in Dukascopy format (e.g., "EURUSD")
            
        Returns:
            Symbol in standard format (e.g., "EUR/USD")
        """
        return cls.REVERSE_INSTRUMENT_MAPPING.get(dukascopy_symbol,
            dukascopy_symbol)

    @classmethod
    def map_to_dukascopy_symbol(cls, standard_symbol: str) ->str:
        """
        Convert standard symbol to Dukascopy format.
        
        Args:
            standard_symbol: Symbol in standard format (e.g., "EUR/USD")
            
        Returns:
            Symbol in Dukascopy format (e.g., "EURUSD")
        """
        return cls.INSTRUMENT_MAPPING.get(standard_symbol, standard_symbol.
            replace('/', ''))

    async def get_tick_data_objects(self, symbol: str, from_time: datetime,
        to_time: datetime, limit: Optional[int]=None) ->List[TickData]:
        """
        Retrieve tick data as TickData objects.
        
        Args:
            symbol: Trading instrument symbol
            from_time: Start time for data query
            to_time: End time for data query
            limit: Maximum number of ticks to return (optional)
            
        Returns:
            List of TickData objects
        """
        tick_dicts = await self.get_tick_data(symbol, from_time, to_time, limit
            )
        return [TickData(symbol=tick['symbol'], timestamp=tick['timestamp'],
            bid=tick['bid'], ask=tick['ask'], bid_volume=tick['bid_volume'],
            ask_volume=tick['ask_volume']) for tick in tick_dicts]

    @async_with_exception_handling
    async def _fetch_data_with_retry(self, url: str, max_retries: int=3,
        backoff_factor: float=1.0) ->Optional[bytes]:
        """
        Fetch data from the given URL with retries on failure.
        
        Args:
            url: The URL to fetch data from
            max_retries: Maximum number of retry attempts
            backoff_factor: Factor by which the delay increases after each retry
            
        Returns:
            Raw data bytes from the response
            
        Raises:
            DataFetchError: If data fetching fails after all retries
        """
        retries = 0
        delay = backoff_factor
        while retries < max_retries:
            try:
                response = await httpx.get(url, timeout=self._session_timeout)
                if response.status_code == 200:
                    return response.content
                elif response.status_code == 404:
                    logger.debug(f'No data found at {url} (404 Not Found)')
                    return None
                else:
                    logger.warning(
                        f'HTTP error {response.status_code} for {url}')
                    return None
            except httpx.TimeoutException as e:
                logger.warning(f'Timeout fetching data from {url}: {e}')
            except httpx.RequestError as e:
                logger.error(f'Request error fetching data from {url}: {e}')
            except Exception as e:
                logger.exception(
                    f'Unexpected error fetching data from {url}: {e}')
            retries += 1
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)
        raise DataFetchError(
            f'Failed to fetch data for {symbol} after {max_retries} retries from {url}'
            , source='Dukascopy', status_code=response.status_code if
            response else None)

    async def _fetch_data(self, symbol: str, start_date: datetime, end_date:
        datetime, timeframe: str) ->Optional[bytes]:
        """
        Fetch historical data from Dukascopy for the given symbol and date range.
        
        Args:
            symbol: The trading instrument symbol
            start_date: The start date and time for the data query
            end_date: The end date and time for the data query
            timeframe: The timeframe for the data (e.g., "1m", "5m", "1h", "1d")
            
        Returns:
            Raw data bytes from Dukascopy, or None if no data is available
        """
        if not self._connected or self._session is None:
            raise DataFetchError('Not connected to Dukascopy')
        start_date_utc = convert_to_utc_datetime(start_date)
        url = (
            f'{self.BASE_URL}/{symbol}/{start_date_utc.year}/{start_date_utc.month - 1:02d}/{start_date_utc.day:02d}/{start_date_utc.hour:02d}h_ticks.bi5'
            )
        raw_data = await self._fetch_data_with_retry(url)
        return raw_data

    def _process_ohlcv_data(self, data: bytes, symbol: str, timeframe: str
        ) ->List[OHLCVData]:
        """
        Process raw OHLCV data bytes into a list of OHLCVData objects.
        
        Args:
            data: Raw data bytes from Dukascopy
            symbol: Trading instrument symbol
            timeframe: Timeframe for the data (e.g., "1m", "5m", "1h", "1d")
            
        Returns:
            List of OHLCVData objects
        """
        logger.warning(
            'OHLCV processing from Dukascopy raw data is not yet implemented.')
        return []

    @with_exception_handling
    def _process_tick_data(self, data: bytes, symbol: str) ->List[TickData]:
        """Processes raw tick data bytes into a list of TickData objects."""
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data), mode='rb') as f:
                decompressed_data = f.read()
            num_ticks = len(decompressed_data) // 20
            if num_ticks == 0:
                return []
            dt_type = np.dtype([('time', 'i4'), ('ask', 'i4'), ('bid', 'i4'
                ), ('ask_vol', 'f4'), ('bid_vol', 'f4')])
            tick_array = np.frombuffer(decompressed_data, dtype=dt_type)
            df = pd.DataFrame(tick_array)
            df['timestamp'] = df['time'].apply(lambda ms: hour_dt +
                timedelta(milliseconds=ms))
            divisor = 1000 if 'JPY' in self.REVERSE_INSTRUMENT_MAPPING.get(
                symbol, '') else 100000
            df['ask'] = df['ask'] / divisor
            df['bid'] = df['bid'] / divisor
            standard_symbol = self.REVERSE_INSTRUMENT_MAPPING.get(symbol,
                symbol)
            ticks = []
            for _, row in df.iterrows():
                ticks.append({'symbol': standard_symbol, 'timestamp': row[
                    'timestamp'].to_pydatetime().replace(tzinfo=timezone.
                    utc), 'bid': row['bid'], 'ask': row['ask'],
                    'bid_volume': row['bid_vol'], 'ask_volume': row['ask_vol']}
                    )
            return ticks
        except (ValueError, IndexError, TypeError) as e:
            self.logger.error(f'Error processing tick data for {symbol}: {e}')
            raise DataTransformationError(
                f'Failed to process tick data for {symbol} due to data format issue: {e}'
                , transformation='Dukascopy Tick Processing') from e
        except Exception as e:
            self.logger.exception(
                f'Unexpected error processing tick data for {symbol}: {e}')
            raise DataTransformationError(
                f'Unexpected error processing tick data for {symbol}: {e}',
                transformation='Dukascopy Tick Processing') from e

    async def get_historical_data(self, symbol: str, from_time: datetime,
        to_time: datetime, timeframe: str='1m', limit: Optional[int]=None
        ) ->List[Union[OHLCVData, TickData]]:
        """
        Retrieve historical data (OHLCV or tick data) for a specific instrument.
        
        Args:
            symbol: Trading instrument symbol
            from_time: Start time for data query
            to_time: End time for data query
            timeframe: Timeframe for the data (e.g., "1m", "5m", "1h", "1d")
            limit: Maximum number of data points to return (optional)
            
        Returns:
            List of historical data objects (OHLCVData or TickData)
        """
        if from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=timezone.utc)
        if to_time.tzinfo is None:
            to_time = to_time.replace(tzinfo=timezone.utc)
        dukascopy_symbol = self.INSTRUMENT_MAPPING.get(symbol)
        if not dukascopy_symbol:
            raise ValueError(f'Unsupported symbol: {symbol}')
        raw_data = await self._fetch_data(dukascopy_symbol, from_time,
            to_time, timeframe)
        if not raw_data:
            logger.warning(
                f'No data returned from Dukascopy for {symbol} from {from_time} to {to_time}'
                )
            return []
        if timeframe in ['1m', '5m', '1h', '1d']:
            ohlcv_data = self._process_ohlcv_data(raw_data,
                dukascopy_symbol, timeframe)
            return ohlcv_data[:limit] if limit else ohlcv_data
        else:
            tick_data = self._process_tick_data(raw_data, dukascopy_symbol)
            return tick_data[:limit] if limit else tick_data
