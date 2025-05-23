"""
Real-Time Data Connector for Causal Inference

This module provides connectors to integrate the causal inference module
with real-time market data sources in the forex trading platform.
"""
import logging
import os
from typing import Dict, List, Optional, Union, Any, Callable
import pandas as pd
import numpy as np
import asyncio
import httpx
from datetime import datetime, timedelta
from analysis_engine.adapters.data_pipeline_adapter import TickDataServiceAdapter
from analysis_engine.causal.data.preparation import FinancialDataPreprocessor, FinancialFeatureEngineering
from analysis_engine.analysis.indicators import IndicatorClient
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class CausalDataConnector:
    """
    Connects real-time and historical market data to the causal inference module.

    This connector integrates with the data pipeline services to provide properly
    formatted data for causal analysis, including real-time streaming data and
    historical datasets from the feature store.
    """

    def __init__(self, indicator_client: IndicatorClient, config: Dict[str,
        Any]=None):
        """
        Initialize the causal data connector.

        Args:
            indicator_client: Client for fetching indicators from the feature-store API.
            config: Configuration parameters for the connector
        """
        self.config = config or {}
        self.tick_data_service = TickDataServiceAdapter(config=self.config)
        feature_store_base_url = self.config.get('feature_store_base_url',
            os.environ.get('FEATURE_STORE_BASE_URL',
            'http://feature-store-service:8000'))
        self.time_series_client = httpx.AsyncClient(base_url=
            f"{feature_store_base_url.rstrip('/')}/api/v1", timeout=30.0)
        self.indicator_client = indicator_client
        self.preprocessor = FinancialDataPreprocessor()
        self.feature_engineering = FinancialFeatureEngineering()
        self.data_cache = {}
        self.last_update = {}
        self.cache_ttl = self.config_manager.get('cache_ttl_minutes', 30)
        self.streaming_callbacks = {}
        self.streaming_tasks = {}
        self.is_streaming = False

    @with_resilience('get_historical_data')
    @async_with_exception_handling
    async def get_historical_data(self, symbols: List[str], start_date:
        datetime, end_date: Optional[datetime]=None, timeframe: str='1h',
        include_indicators: bool=True, force_refresh: bool=False
        ) ->pd.DataFrame:
        """
        Get historical market data for causal analysis.

        Args:
            symbols: List of forex symbols to fetch (e.g., ['EURUSD', 'GBPUSD'])
            start_date: Start date for the historical data
            end_date: End date for the historical data (defaults to now)
            timeframe: Data timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            include_indicators: Whether to include technical indicators
            force_refresh: Whether to force a refresh of cached data

        Returns:
            DataFrame with prepared historical data
        """
        end_date = end_date or datetime.now()
        symbols_key = '_'.join(sorted(symbols))
        cache_key = (
            f'{symbols_key}_{timeframe}_{start_date.date()}_{end_date.date()}')
        if not force_refresh and cache_key in self.data_cache:
            last_update = self.last_update.get(cache_key, datetime.min)
            if datetime.now() - last_update < timedelta(minutes=self.cache_ttl
                ):
                logger.info(f'Using cached data for {cache_key}')
                return self.data_cache[cache_key]
        all_data = {}
        for symbol in symbols:
            try:
                response = await self.time_series_client.get(
                    '/time-series/historical', params={'symbol': symbol,
                    'start_date': start_date.isoformat(), 'end_date':
                    end_date.isoformat(), 'timeframe': timeframe})
                response.raise_for_status()
                data = response.json()
                symbol_data = pd.DataFrame(data['data'])
                if not symbol_data.empty:
                    symbol_data['timestamp'] = pd.to_datetime(symbol_data[
                        'timestamp'])
                    symbol_data.set_index('timestamp', inplace=True)
                else:
                    symbol_data = pd.DataFrame()
            except httpx.HTTPStatusError as e:
                logger.error(
                    f'HTTP error occurred: {e.response.status_code} - {e.response.text}'
                    )
                symbol_data = None
            except httpx.RequestError as e:
                logger.error(
                    f'Request error while getting historical data for {symbol}: {str(e)}'
                    )
                symbol_data = None
            if symbol_data is not None:
                symbol_data.columns = [f'{symbol}_{col}' for col in
                    symbol_data.columns]
                all_data[symbol] = symbol_data
        if not all_data:
            logger.error('No historical data retrieved for symbols')
            return pd.DataFrame()
        combined_data = pd.concat(all_data.values(), axis=1)
        if include_indicators:
            for symbol in symbols:
                price_col = f'{symbol}_close'
                volume_col = (f'{symbol}_volume' if f'{symbol}_volume' in
                    combined_data.columns else None)
                if price_col in combined_data.columns:
                    symbol_subset = {'price': combined_data[price_col]}
                    if volume_col:
                        symbol_subset['volume'] = combined_data[volume_col]
                    symbol_df = pd.DataFrame(symbol_subset)
                    with_indicators = await self._add_indicators_via_api(
                        symbol_df, symbol=symbol, timeframe=timeframe,
                        start_date=start_date, end_date=end_date)
                    for col in with_indicators.columns:
                        if col in ['price', 'volume']:
                            continue
                        combined_data[f'{symbol}_{col}'] = with_indicators[col]
        self.data_cache[cache_key] = combined_data
        self.last_update[cache_key] = datetime.now()
        return combined_data

    @with_resilience('get_real_time_data')
    async def get_real_time_data(self, symbols: List[str], window_size:
        timedelta=timedelta(hours=24), timeframe: str='5m',
        include_indicators: bool=True) ->pd.DataFrame:
        """
        Get the most recent market data for real-time causal analysis.

        Args:
            symbols: List of forex symbols to fetch
            window_size: Time window to fetch data for
            timeframe: Data timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            include_indicators: Whether to include technical indicators

        Returns:
            DataFrame with prepared real-time data
        """
        start_date = datetime.now() - window_size
        return await self.get_historical_data(symbols=symbols, start_date=
            start_date, end_date=datetime.now(), timeframe=timeframe,
            include_indicators=include_indicators, force_refresh=True)

    @async_with_exception_handling
    async def start_streaming(self, symbols: List[str], callback: Callable[
        [pd.DataFrame], None], interval: int=60, timeframe: str='1m',
        window_size: timedelta=timedelta(hours=4), include_indicators: bool
        =True) ->str:
        """
        Start streaming real-time data updates for causal analysis.

        Args:
            symbols: List of forex symbols to stream
            callback: Function to call with updated data
            interval: Streaming interval in seconds
            timeframe: Data timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            window_size: Time window of data to provide
            include_indicators: Whether to include technical indicators

        Returns:
            Stream ID for stopping the stream later
        """
        stream_id = (
            f'stream_{len(self.streaming_callbacks) + 1}_{datetime.now().timestamp()}'
            )
        self.streaming_callbacks[stream_id] = callback

        @async_with_exception_handling
        async def streaming_task():
    """
    Streaming task.
    
    """

            while stream_id in self.streaming_callbacks:
                try:
                    data = await self.get_real_time_data(symbols=symbols,
                        window_size=window_size, timeframe=timeframe,
                        include_indicators=include_indicators)
                    if stream_id in self.streaming_callbacks:
                        self.streaming_callbacks[stream_id](data)
                except Exception as e:
                    logger.error(
                        f'Error in streaming task for {stream_id}: {str(e)}')
                await asyncio.sleep(interval)
        task = asyncio.create_task(streaming_task())
        self.streaming_tasks[stream_id] = task
        self.is_streaming = True
        logger.info(f'Started streaming data for {symbols} with ID {stream_id}'
            )
        return stream_id

    async def stop_streaming(self, stream_id: str) ->bool:
        """
        Stop a streaming data connection.

        Args:
            stream_id: ID of the stream to stop

        Returns:
            True if the stream was successfully stopped, False otherwise
        """
        if stream_id in self.streaming_callbacks:
            del self.streaming_callbacks[stream_id]
            if stream_id in self.streaming_tasks:
                self.streaming_tasks[stream_id].cancel()
                del self.streaming_tasks[stream_id]
            logger.info(f'Stopped streaming data for stream {stream_id}')
            self.is_streaming = len(self.streaming_callbacks) > 0
            return True
        logger.warning(f'Stream {stream_id} not found')
        return False

    async def prepare_data_for_causal_analysis(self, data: pd.DataFrame,
        make_stationary: bool=True, scale_data: bool=True) ->pd.DataFrame:
        """
        Prepare data specifically for causal analysis.

        Args:
            data: Raw market data
            make_stationary: Whether to make the data stationary
            scale_data: Whether to scale the data

        Returns:
            DataFrame with data prepared for causal analysis
        """
        prepared_data = data.copy()
        if make_stationary:
            prepared_data = self.preprocessor.prepare_data(prepared_data)
        if scale_data:
            prepared_data = self.preprocessor.scale_data(prepared_data)
        return prepared_data

    @async_with_exception_handling
    async def _add_indicators_via_api(self, df: pd.DataFrame, symbol: str,
        timeframe: str, start_date: datetime, end_date: datetime
        ) ->pd.DataFrame:
        """
        Add technical indicators to the dataframe by fetching them from the feature-store-service API.

        Args:
            df: DataFrame with price and volume data
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with added indicators
        """
        indicators_to_fetch = ['sma', 'ema', 'rsi', 'macd',
            'bollinger_bands', 'atr']
        result_df = df.copy()
        for indicator_name in indicators_to_fetch:
            try:
                indicator_df = (await self.indicator_client.
                    get_indicator_values(name=indicator_name, symbol=symbol,
                    timeframe=timeframe, start_date=start_date, end_date=
                    end_date))
                if not indicator_df.empty:
                    indicator_df.columns = [f'{indicator_name}_{col}' for
                        col in indicator_df.columns]
                    result_df = result_df.join(indicator_df, how='left')
            except Exception as e:
                logger.warning(
                    f'Failed to fetch indicator {indicator_name}: {str(e)}')
        return result_df
