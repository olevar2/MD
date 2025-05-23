"""
Time Series Data Service

This module provides high-performance data retrieval services for time series data,
leveraging the TimeSeriesQueryOptimizer for efficient database access.
"""
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
from functools import lru_cache
from core.time_series_query_optimizer_1 import TimeSeriesQueryOptimizer, TimeSeriesQueryContext
from feature_store_service.db.connection import get_db_connection
from feature_store_service.models.time_series import TimeSeriesData


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class TimeSeriesDataService:
    """
    Service for efficient retrieval and management of time series data,
    with optimized query handling and caching.
    """

    def __init__(self, max_cache_size: int=200, default_ttl: int=600):
        """
        Initialize the time series data service

        Args:
            max_cache_size: Maximum number of query results to cache
            default_ttl: Default time-to-live for cached results in seconds
        """
        self.logger = logging.getLogger(__name__)
        self.optimizer = TimeSeriesQueryOptimizer(max_cache_size=
            max_cache_size, default_ttl=default_ttl)

    async def get_price_data(self, symbol: str, timeframe: str, start_time:
        Union[datetime, str], end_time: Union[datetime, str],
        include_current_candle: bool=False) ->pd.DataFrame:
        """
        Get historical price data with optimization for repeated queries

        Args:
            symbol: Trading instrument symbol
            timeframe: Chart timeframe (e.g., '1m', '5m', '1h', '1d')
            start_time: Start of the time range
            end_time: End of the time range
            include_current_candle: Whether to include the current (incomplete) candle

        Returns:
            DataFrame with OHLCV data
        """
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
        query_params = {'symbol': symbol, 'timeframe': timeframe,
            'start_time': start_time, 'end_time': end_time,
            'include_current_candle': include_current_candle}
        with TimeSeriesQueryContext(self.optimizer, query_params) as (cache_hit
            , result, optimized_params):
            if cache_hit:
                self.logger.debug(f'Cache hit for {symbol} {timeframe} data')
                return result
            db = get_db_connection()
            db_query_params = self._prepare_db_query_params(optimized_params)
            data = await self._execute_db_query(db, db_query_params)
            self.optimizer.cache_result(optimized_params, data)
            return data

    async def get_indicator_data(self, symbol: str, indicator_name: str,
        parameters: Dict[str, Any], timeframe: str, start_time: Union[
        datetime, str], end_time: Union[datetime, str]) ->pd.DataFrame:
        """
        Get technical indicator data with optimization for repeated queries

        Args:
            symbol: Trading instrument symbol
            indicator_name: Name of the technical indicator
            parameters: Dictionary of indicator parameters
            timeframe: Chart timeframe
            start_time: Start of the time range
            end_time: End of the time range

        Returns:
            DataFrame with indicator data
        """
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
        query_params = {'symbol': symbol, 'indicator_name': indicator_name,
            'indicator_params': parameters, 'timeframe': timeframe,
            'start_time': start_time, 'end_time': end_time}
        with TimeSeriesQueryContext(self.optimizer, query_params) as (cache_hit
            , result, optimized_params):
            if cache_hit:
                self.logger.debug(
                    f'Cache hit for {indicator_name} on {symbol} {timeframe}')
                return result
            db = get_db_connection()
            db_query_params = self._prepare_db_query_params(optimized_params)
            data = await self._execute_indicator_query(db, db_query_params)
            if data.empty:
                price_data = await self.get_price_data(symbol=symbol,
                    timeframe=timeframe, start_time=optimized_params[
                    'start_time'], end_time=optimized_params['end_time'])
                data = await self._calculate_indicator(price_data=
                    price_data, indicator_name=indicator_name, parameters=
                    parameters)
                self._schedule_indicator_storage(symbol=symbol, timeframe=
                    timeframe, indicator_name=indicator_name, parameters=
                    parameters, data=data)
            self.optimizer.cache_result(optimized_params, data)
            return data

    @async_with_exception_handling
    async def get_multi_symbol_data(self, symbols: List[str], timeframe:
        str, start_time: Union[datetime, str], end_time: Union[datetime,
        str], align_timestamps: bool=True, alignment_method: str='ffill',
        alignment_limit: Optional[int]=None, use_common_range: bool=False,
        parallel_fetch: bool=True, batch_size: int=5) ->Dict[str, pd.DataFrame
        ]:
        """
        Get price data for multiple symbols efficiently with optimized parallel processing

        Args:
            symbols: List of trading instrument symbols
            timeframe: Chart timeframe
            start_time: Start of the time range
            end_time: End of the time range
            align_timestamps: Whether to align timestamps across all symbols
            alignment_method: Fill method for missing values ('ffill', 'bfill', 'nearest', None)
            alignment_limit: Maximum number of consecutive NaN values to fill
            use_common_range: If True, only use the time range common to all symbols
            parallel_fetch: Whether to fetch data in parallel
            batch_size: Number of symbols to process in each parallel batch

        Returns:
            Dictionary mapping symbols to their respective DataFrames
        """
        if not symbols:
            return {}
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
        cache_key = None
        if align_timestamps:
            symbols_str = '_'.join(sorted(symbols))
            time_range_str = f'{start_time.isoformat()}_{end_time.isoformat()}'
            cache_key = (
                f'align_{symbols_str}_{timeframe}_{time_range_str}_{use_common_range}'
                )
        results = {}
        if parallel_fetch and len(symbols) > 1:
            symbol_batches = [symbols[i:i + batch_size] for i in range(0,
                len(symbols), batch_size)]
            import asyncio

            @async_with_exception_handling
            async def fetch_batch(batch):
    """
    Fetch batch.
    
    Args:
        batch: Description of batch
    
    """

                batch_results = {}
                fetch_tasks = []
                for symbol in batch:
                    task = self.get_price_data(symbol=symbol, timeframe=
                        timeframe, start_time=start_time, end_time=end_time)
                    fetch_tasks.append((symbol, task))
                for symbol, task in fetch_tasks:
                    try:
                        data = await task
                        batch_results[symbol] = data
                    except Exception as e:
                        self.logger.error(
                            f'Error fetching data for {symbol}: {str(e)}')
                        batch_results[symbol] = pd.DataFrame()
                return batch_results
            batch_tasks = [fetch_batch(batch) for batch in symbol_batches]
            batch_results = await asyncio.gather(*batch_tasks)
            for batch_result in batch_results:
                results.update(batch_result)
        else:
            for symbol in symbols:
                try:
                    data = await self.get_price_data(symbol=symbol,
                        timeframe=timeframe, start_time=start_time,
                        end_time=end_time)
                    results[symbol] = data
                except Exception as e:
                    self.logger.error(
                        f'Error fetching data for {symbol}: {str(e)}')
                    results[symbol] = pd.DataFrame()
        if align_timestamps and results:
            results = self._align_multi_symbol_data(results, method=
                alignment_method, limit=alignment_limit, use_common_range=
                use_common_range, cache_key=cache_key)
        return results

    def invalidate_cache(self, symbol: Optional[str]=None, indicator_type:
        Optional[str]=None, timeframe: Optional[str]=None):
        """
        Invalidate cache entries for specific symbols, indicators or timeframes

        Args:
            symbol: Symbol to invalidate cache for
            indicator_type: Indicator type to invalidate cache for
            timeframe: Timeframe to invalidate cache for
        """
        self.optimizer.invalidate_cache(symbol=symbol, indicator_type=
            indicator_type)
        self.logger.info(
            f"Invalidated cache for {symbol or 'all symbols'}, {indicator_type or 'all indicators'}, {timeframe or 'all timeframes'}"
            )

    def get_optimizer_statistics(self) ->Dict[str, Any]:
        """
        Get statistics about the query optimizer performance

        Returns:
            Dictionary with optimizer statistics
        """
        return self.optimizer.get_query_statistics()

    @async_with_exception_handling
    async def _execute_db_query(self, db, query_params: Dict[str, Any]
        ) ->pd.DataFrame:
        """
        Execute a database query for price data

        Args:
            db: Database connection
            query_params: Query parameters

        Returns:
            DataFrame with query results
        """
        try:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM price_data
                WHERE symbol = :symbol
                AND timeframe = :timeframe
                AND timestamp >= :start_time
                AND timestamp <= :end_time
                ORDER BY timestamp
            """
            if query_params.get('time_index'):
                query = query.replace('FROM price_data',
                    'FROM price_data USING INDEX time_idx')
            if query_params.get('symbol_index'):
                query = query.replace('FROM price_data',
                    'FROM price_data USING INDEX symbol_idx')
            result = await db.fetch_all(query, query_params)
            if result:
                data = pd.DataFrame(result)
                data.set_index('timestamp', inplace=True)
                return data
            else:
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f'Database query error: {str(e)}')
            return pd.DataFrame()

    @async_with_exception_handling
    async def _execute_indicator_query(self, db, query_params: Dict[str, Any]
        ) ->pd.DataFrame:
        """
        Execute a database query for indicator data

        Args:
            db: Database connection
            query_params: Query parameters

        Returns:
            DataFrame with indicator data
        """
        try:
            query = """
                SELECT timestamp, value, additional_values
                FROM indicator_data
                WHERE symbol = :symbol
                AND indicator_name = :indicator_name
                AND parameters = :indicator_params_json
                AND timeframe = :timeframe
                AND timestamp >= :start_time
                AND timestamp <= :end_time
                ORDER BY timestamp
            """
            import json
            query_params['indicator_params_json'] = json.dumps(query_params
                ['indicator_params'], sort_keys=True)
            result = await db.fetch_all(query, query_params)
            if result:
                data = pd.DataFrame(result)
                if 'additional_values' in data.columns:
                    additional_df = pd.json_normalize(data[
                        'additional_values'].apply(json.loads))
                    data = pd.concat([data.drop('additional_values', axis=1
                        ), additional_df], axis=1)
                data.set_index('timestamp', inplace=True)
                return data
            else:
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f'Indicator query error: {str(e)}')
            return pd.DataFrame()

    @async_with_exception_handling
    async def _calculate_indicator(self, price_data: pd.DataFrame,
        indicator_name: str, parameters: Dict[str, Any]) ->pd.DataFrame:
        """
        Calculate a technical indicator from price data

        Args:
            price_data: OHLCV price data
            indicator_name: Name of the indicator to calculate
            parameters: Dictionary of indicator parameters

        Returns:
            DataFrame with indicator values
        """
        try:
            from core.factory import create_indicator
            indicator = create_indicator(indicator_name, parameters)
            indicator_data = indicator.calculate(price_data)
            return indicator_data
        except Exception as e:
            self.logger.error(f'Error calculating {indicator_name}: {str(e)}')
            return pd.DataFrame()

    def _prepare_db_query_params(self, query_params: Dict[str, Any]) ->Dict[
        str, Any]:
        """
        Prepare query parameters for database queries

        Args:
            query_params: Original query parameters

        Returns:
            Prepared parameters for database query
        """
        db_params = query_params.copy()
        if isinstance(db_params.get('start_time'), datetime):
            db_params['start_time'] = db_params['start_time'].isoformat()
        if isinstance(db_params.get('end_time'), datetime):
            db_params['end_time'] = db_params['end_time'].isoformat()
        return db_params

    @with_exception_handling
    def _schedule_indicator_storage(self, symbol: str, timeframe: str,
        indicator_name: str, parameters: Dict[str, Any], data: pd.DataFrame):
        """
        Schedule background task to store calculated indicator values in database

        Args:
            symbol: Trading instrument symbol
            timeframe: Chart timeframe
            indicator_name: Name of the indicator
            parameters: Dictionary of indicator parameters
            data: DataFrame with indicator values
        """
        try:
            from feature_store_service.tasks.storage_tasks import store_indicator_values
            records = data.reset_index().to_dict('records')
            store_indicator_values.delay(symbol=symbol, timeframe=timeframe,
                indicator_name=indicator_name, parameters=parameters,
                values=records)
        except Exception as e:
            self.logger.error(f'Error scheduling indicator storage: {str(e)}')

    def _align_multi_symbol_data(self, symbol_data: Dict[str, pd.DataFrame],
        method: str='ffill', limit: Optional[int]=None, use_common_range:
        bool=False, cache_key: Optional[str]=None) ->Dict[str, pd.DataFrame]:
        """
        Align timestamp indexes across multiple symbol DataFrames with optimized performance

        Args:
            symbol_data: Dictionary mapping symbols to their DataFrames
            method: Fill method for missing values ('ffill', 'bfill', 'nearest', None)
            limit: Maximum number of consecutive NaN values to fill
            use_common_range: If True, only use the time range common to all symbols
            cache_key: Optional cache key for storing/retrieving alignment results

        Returns:
            Dictionary with aligned DataFrames
        """
        if not symbol_data:
            return {}
        if cache_key and hasattr(self, '_alignment_cache'):
            if cache_key in self._alignment_cache:
                self.logger.debug(f'Alignment cache hit for {cache_key}')
                return self._alignment_cache[cache_key]
        elif not hasattr(self, '_alignment_cache'):
            self._alignment_cache = {}
        non_empty_data = {s: df for s, df in symbol_data.items() if not df.
            empty}
        if not non_empty_data:
            return symbol_data.copy()
        if use_common_range and len(non_empty_data) > 1:
            min_timestamps = [df.index.min() for df in non_empty_data.values()]
            max_timestamps = [df.index.max() for df in non_empty_data.values()]
            common_start = max(min_timestamps)
            common_end = min(max_timestamps)
            sampling_intervals = []
            for df in non_empty_data.values():
                if len(df) > 1:
                    intervals = df.index.to_series().diff().dropna().median()
                    sampling_intervals.append(intervals)
            if sampling_intervals:
                interval = pd.Timedelta(seconds=int(pd.Series(
                    sampling_intervals).median().total_seconds()))
                all_timestamps = pd.date_range(start=common_start, end=
                    common_end, freq=interval)
            else:
                all_timestamps = pd.Index([])
                for df in non_empty_data.values():
                    mask = (df.index >= common_start) & (df.index <= common_end
                        )
                    all_timestamps = all_timestamps.union(df.index[mask])
        else:
            all_timestamps = pd.Index([])
            for df in non_empty_data.values():
                all_timestamps = all_timestamps.union(df.index)
        aligned_data = {}
        if len(symbol_data) <= 5 or len(all_timestamps) < 10000:
            for symbol, df in symbol_data.items():
                if df.empty:
                    aligned_data[symbol] = df
                else:
                    aligned_df = df.reindex(all_timestamps)
                    if method:
                        aligned_df = aligned_df.fillna(method=method, limit
                            =limit)
                    aligned_data[symbol] = aligned_df
        else:
            import concurrent.futures

            def align_single_df(symbol_df_tuple):
    """
    Align single df.
    
    Args:
        symbol_df_tuple: Description of symbol_df_tuple
    
    """

                symbol, df = symbol_df_tuple
                if df.empty:
                    return symbol, df
                else:
                    aligned_df = df.reindex(all_timestamps)
                    if method:
                        aligned_df = aligned_df.fillna(method=method, limit
                            =limit)
                    return symbol, aligned_df
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(align_single_df, symbol_data.items())
                for symbol, aligned_df in results:
                    aligned_data[symbol] = aligned_df
        if cache_key:
            self._alignment_cache[cache_key] = aligned_data
            if len(self._alignment_cache) > 100:
                oldest_keys = list(self._alignment_cache.keys())[:-100]
                for key in oldest_keys:
                    del self._alignment_cache[key]
        return aligned_data
