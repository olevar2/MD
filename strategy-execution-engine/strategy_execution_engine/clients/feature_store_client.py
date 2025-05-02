"""
Feature Store Client Module

This module provides a client to interact with the feature store service,
for retrieving OHLCV data, technical indicators, and other features.
"""
import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import os
import time
from functools import lru_cache

from core_foundations.utils.logger import get_logger
from core_foundations.resilience.circuit_breaker import CircuitBreaker
from core_foundations.resilience.retry import retry_with_backoff
from core_foundations.exceptions.client_exceptions import (
    FeatureStoreConnectionError,
    FeatureStoreTimeoutError,
    FeatureStoreAuthError,
    FeatureNotFoundError
)

# Import optimized JSON parsing
from strategy_execution_engine.clients.json_optimized import parse_json_response, dumps, loads

# Import metrics collector if available
try:
    from strategy_execution_engine.monitoring.feature_store_metrics import feature_store_metrics
    has_metrics = True
except ImportError:
    has_metrics = False

# Import caching module if available
try:
    from strategy_execution_engine.caching.feature_cache import FeatureCache
    has_feature_cache = True
except ImportError:
    has_feature_cache = False

# Try to import ujson for faster JSON parsing
try:
    import ujson as json
    has_ujson = True
except ImportError:
    import json
    has_ujson = False

# Try to import orjson for even faster JSON parsing
try:
    import orjson
    has_orjson = True
except ImportError:
    has_orjson = False


class FeatureStoreClient:
    """
    A client for interacting with the feature store service.

    This class provides methods to retrieve OHLCV data, technical indicators,
    and other features from the feature store service.

    Attributes:
        base_url: Base URL of the feature store service
        headers: HTTP headers for API requests
        logger: Logger instance
        circuit_breaker: Circuit breaker for fault tolerance
        cache: Feature cache instance (if available)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        use_cache: bool = True,
        cache_ttl: int = 300,  # 5 minutes default TTL
        timeout: float = 30.0,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_recovery_time: int = 60
    ):
        """
        Initialize the FeatureStoreClient with connection settings.

        Args:
            base_url: Base URL of the feature store service API.
                     If None, uses FEATURE_STORE_URL environment variable or default.
            api_key: API key for authentication.
                    If None, uses FEATURE_STORE_API_KEY environment variable.
            use_cache: Whether to use caching for feature store results
            cache_ttl: Time-to-live for cached results in seconds
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            circuit_breaker_threshold: Number of failures before circuit breaker opens
            circuit_breaker_recovery_time: Time in seconds before circuit breaker resets
        """
        self.base_url = base_url or os.getenv('FEATURE_STORE_URL', 'http://localhost:8001/api/v1')
        api_key = api_key or os.getenv('FEATURE_STORE_API_KEY')
        self.timeout = timeout
        self.max_retries = max_retries

        # Set up headers for API requests
        self.headers = {
            'Content-Type': 'application/json'
        }

        # Add API key to headers if available
        if api_key:
            self.headers['X-API-Key'] = api_key

        # Set up logger
        self.logger = get_logger("feature_store_client")

        # Set up circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            recovery_timeout=circuit_breaker_recovery_time,
            name="feature_store_client"
        )

        # Set up caching if available and enabled
        self.use_cache = use_cache and has_feature_cache
        self.cache_ttl = cache_ttl
        self.cache = FeatureCache() if self.use_cache else None

        # Initialize async HTTP session
        self._session = None

        self.logger.info(f"FeatureStoreClient initialized with base URL: {self.base_url}")
        if self.use_cache:
            self.logger.info(f"Caching enabled with TTL: {cache_ttl} seconds")

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    @retry_with_backoff(
        retries=3,
        backoff_factor=1.5,
        exceptions=(
            requests.RequestException,
            requests.Timeout,
            requests.ConnectionError,
            aiohttp.ClientError,
            asyncio.TimeoutError
        )
    )
    async def get_ohlcv_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str = '1h'
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV data from the feature store.

        Args:
            symbol: Trading symbol (e.g., 'EUR_USD')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe for the data (e.g., '1m', '5m', '1h', '1d')

        Returns:
            pd.DataFrame: DataFrame containing OHLCV data

        Raises:
            FeatureStoreConnectionError: If connection to feature store fails
            FeatureStoreTimeoutError: If request times out
            FeatureStoreAuthError: If authentication fails
        """
        # Check circuit breaker
        if self.circuit_breaker.is_open:
            self.logger.warning("Circuit breaker is open, using fallback data")
            return self._get_fallback_ohlcv_data(symbol, start_date, end_date, timeframe)

        try:
            # Format dates
            start_date_str = self._format_date(start_date)
            end_date_str = self._format_date(end_date)

            # Check cache first if enabled
            if self.use_cache:
                cache_key = f"ohlcv_{symbol}_{timeframe}_{start_date_str}_{end_date_str}"
                cached_data = self.cache.get(cache_key)
                if cached_data is not None:
                    self.logger.debug(f"Cache hit for {cache_key}")
                    return cached_data

            # Build URL and parameters
            url = f"{self.base_url}/data/ohlcv"
            params = {
                "symbol": symbol,
                "start_date": start_date_str,
                "end_date": end_date_str,
                "timeframe": timeframe
            }

            self.logger.info(f"Requesting OHLCV data for {symbol} from {start_date_str} to {end_date_str} ({timeframe})")

            # Make the request
            session = await self.get_session()
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await parse_json_response(response)

                    # Convert to DataFrame
                    if 'data' in data and data['data']:
                        df = pd.DataFrame(data['data'])

                        # Convert timestamp to datetime
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])

                        # Cache the result if caching is enabled
                        if self.use_cache:
                            self.cache.set(cache_key, df, ttl=self.cache_ttl)

                        self.logger.info(f"Retrieved {len(df)} OHLCV records")
                        return df
                    else:
                        self.logger.warning(f"No OHLCV data returned for {symbol}")
                        return pd.DataFrame()
                elif response.status == 401 or response.status == 403:
                    raise FeatureStoreAuthError(f"Authentication failed: {response.status}")
                else:
                    error_text = await response.text()
                    raise FeatureStoreConnectionError(f"Error fetching OHLCV data: {error_text}")

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            self.circuit_breaker.record_failure()
            self.logger.error(f"Connection error: {str(e)}")
            raise FeatureStoreConnectionError(f"Connection error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error retrieving OHLCV data: {str(e)}")
            raise

    @retry_with_backoff(
        retries=3,
        backoff_factor=1.5,
        exceptions=(
            requests.RequestException,
            requests.Timeout,
            requests.ConnectionError,
            aiohttp.ClientError,
            asyncio.TimeoutError
        )
    )
    async def get_indicators(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str = '1h',
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Wrapper for _get_indicators with metrics tracking."""
        start_time = time.time()

        try:
            # Track API call if metrics are available
            if has_metrics:
                feature_store_metrics.metrics["api_calls"]["total"] += 1
                feature_store_metrics.metrics["api_calls"]["get_indicators"] += 1

            # Call the actual implementation
            result = await self._get_indicators_impl(symbol, start_date, end_date, timeframe, indicators)

            # Track performance metrics if available
            if has_metrics:
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                feature_store_metrics.metrics["performance"]["request_count"] += 1
                feature_store_metrics.metrics["performance"]["total_response_time_ms"] += response_time_ms
                feature_store_metrics.metrics["performance"]["avg_response_time_ms"] = (
                    feature_store_metrics.metrics["performance"]["total_response_time_ms"] /
                    feature_store_metrics.metrics["performance"]["request_count"]
                )

            return result
        except Exception as e:
            # Track error metrics if available
            if has_metrics:
                feature_store_metrics.track_error(str(type(e).__name__))
            raise

    async def _get_indicators_impl(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str = '1h',
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Retrieve technical indicators from the feature store.

        Args:
            symbol: Trading symbol (e.g., 'EUR_USD')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe for the data
            indicators: List of indicator names to retrieve

        Returns:
            pd.DataFrame: DataFrame containing indicator values

        Raises:
            FeatureStoreConnectionError: If connection to feature store fails
            FeatureStoreTimeoutError: If request times out
            FeatureStoreAuthError: If authentication fails
        """
        # Check circuit breaker
        if self.circuit_breaker.is_open:
            self.logger.warning("Circuit breaker is open, using fallback data")
            # Track fallback if metrics are available
            if has_metrics:
                feature_store_metrics.track_fallback("get_indicators")
            return self._get_fallback_indicator_data(symbol, start_date, end_date, timeframe, indicators)

        try:
            # Format dates
            start_date_str = self._format_date(start_date)
            end_date_str = self._format_date(end_date)

            # Check cache first if enabled
            if self.use_cache and indicators:
                indicators_key = ",".join(sorted(indicators))
                cache_key = f"indicators_{symbol}_{timeframe}_{indicators_key}_{start_date_str}_{end_date_str}"
                cached_data = self.cache.get(cache_key)
                if cached_data is not None:
                    self.logger.debug(f"Cache hit for {cache_key}")
                    # Track cache hit if metrics are available
                    if has_metrics:
                        feature_store_metrics.track_cache_hit()
                    return cached_data
                else:
                    # Track cache miss if metrics are available
                    if has_metrics:
                        feature_store_metrics.track_cache_miss()

            # Build URL and parameters
            url = f"{self.base_url}/features/indicators"
            params = {
                "symbol": symbol,
                "start_date": start_date_str,
                "end_date": end_date_str,
                "timeframe": timeframe
            }

            # Add indicators to parameters if specified
            if indicators:
                params["indicators"] = ",".join(indicators)

            self.logger.info(f"Requesting indicators for {symbol} from {start_date_str} to {end_date_str} ({timeframe})")
            if indicators:
                self.logger.info(f"Requested indicators: {indicators}")

            # Make the request
            session = await self.get_session()
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await parse_json_response(response)

                    # Convert to DataFrame
                    if 'data' in data and data['data']:
                        df = pd.DataFrame(data['data'])

                        # Convert timestamp to datetime
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])

                        # Cache the result if caching is enabled
                        if self.use_cache and indicators:
                            self.cache.set(cache_key, df, ttl=self.cache_ttl)

                        self.logger.info(f"Retrieved {len(df)} indicator records")
                        return df
                    else:
                        self.logger.warning(f"No indicator data returned for {symbol}")
                        return pd.DataFrame()
                elif response.status == 401 or response.status == 403:
                    raise FeatureStoreAuthError(f"Authentication failed: {response.status}")
                else:
                    error_text = await response.text()
                    raise FeatureStoreConnectionError(f"Error fetching indicators: {error_text}")

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            self.circuit_breaker.record_failure()
            self.logger.error(f"Connection error: {str(e)}")
            raise FeatureStoreConnectionError(f"Connection error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error retrieving indicators: {str(e)}")
            raise

    @retry_with_backoff(
        retries=3,
        backoff_factor=1.5,
        exceptions=(
            requests.RequestException,
            requests.Timeout,
            requests.ConnectionError,
            aiohttp.ClientError,
            asyncio.TimeoutError
        )
    )
    async def compute_feature(
        self,
        feature_name: str,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str = '1h',
        parameters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Request computation of a specific feature.

        Args:
            feature_name: Name of the feature to compute
            symbol: Trading symbol
            start_date: Start date for computation
            end_date: End date for computation
            timeframe: Timeframe for the data
            parameters: Optional parameters for feature computation

        Returns:
            pd.DataFrame: DataFrame containing computed feature values

        Raises:
            FeatureStoreConnectionError: If connection to feature store fails
            FeatureStoreTimeoutError: If request times out
            FeatureStoreAuthError: If authentication fails
            FeatureNotFoundError: If the requested feature is not found
        """
        # Check circuit breaker
        if self.circuit_breaker.is_open:
            self.logger.warning("Circuit breaker is open, using fallback data")
            return self._get_fallback_feature_data(feature_name, symbol, start_date, end_date, timeframe)

        try:
            # Format dates
            start_date_str = self._format_date(start_date)
            end_date_str = self._format_date(end_date)

            # Check cache first if enabled
            if self.use_cache:
                params_key = dumps(parameters or {}, sort_keys=True)
                cache_key = f"feature_{feature_name}_{symbol}_{timeframe}_{params_key}_{start_date_str}_{end_date_str}"
                cached_data = self.cache.get(cache_key)
                if cached_data is not None:
                    self.logger.debug(f"Cache hit for {cache_key}")
                    return cached_data

            # Build URL and request body
            url = f"{self.base_url}/features/compute"
            body = {
                "feature_name": feature_name,
                "symbol": symbol,
                "start_date": start_date_str,
                "end_date": end_date_str,
                "timeframe": timeframe
            }

            # Add parameters if provided
            if parameters:
                body["parameters"] = parameters

            self.logger.info(f"Requesting computation of feature '{feature_name}' for {symbol}")

            # Make the POST request
            session = await self.get_session()
            async with session.post(url, json=body) as response:
                if response.status == 200:
                    data = await parse_json_response(response)

                    # Convert to DataFrame
                    if 'data' in data and data['data']:
                        df = pd.DataFrame(data['data'])

                        # Convert timestamp to datetime
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])

                        # Cache the result if caching is enabled
                        if self.use_cache:
                            self.cache.set(cache_key, df, ttl=self.cache_ttl)

                        self.logger.info(f"Retrieved {len(df)} computed feature records")
                        return df
                    else:
                        self.logger.warning(f"No computed feature data returned for {feature_name}")
                        return pd.DataFrame()
                elif response.status == 404:
                    raise FeatureNotFoundError(f"Feature '{feature_name}' not found")
                elif response.status == 401 or response.status == 403:
                    raise FeatureStoreAuthError(f"Authentication failed: {response.status}")
                else:
                    error_text = await response.text()
                    raise FeatureStoreConnectionError(f"Error computing feature: {error_text}")

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            self.circuit_breaker.record_failure()
            self.logger.error(f"Connection error: {str(e)}")
            raise FeatureStoreConnectionError(f"Connection error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error computing feature: {str(e)}")
            raise

    async def get_available_indicators(self) -> List[Dict[str, Any]]:
        """
        Get a list of all available indicators in the feature store.

        Returns:
            List of dictionaries containing indicator metadata

        Raises:
            FeatureStoreConnectionError: If connection to feature store fails
        """
        try:
            # Check cache first if enabled
            if self.use_cache:
                cache_key = "available_indicators"
                cached_data = self.cache.get(cache_key)
                if cached_data is not None:
                    return cached_data

            # Build URL
            url = f"{self.base_url}/indicators"

            # Make the request
            session = await self.get_session()
            async with session.get(url) as response:
                if response.status == 200:
                    data = await parse_json_response(response)
                    indicators = data.get('indicators', [])

                    # Cache the result if caching is enabled
                    if self.use_cache:
                        self.cache.set(cache_key, indicators, ttl=self.cache_ttl * 2)  # Longer TTL for metadata

                    return indicators
                else:
                    error_text = await response.text()
                    raise FeatureStoreConnectionError(f"Error fetching available indicators: {error_text}")

        except Exception as e:
            self.logger.error(f"Error retrieving available indicators: {str(e)}")
            raise FeatureStoreConnectionError(f"Error retrieving available indicators: {str(e)}")

    async def get_indicator_metadata(self, indicator_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific indicator.

        Args:
            indicator_id: ID of the indicator

        Returns:
            Dictionary containing indicator metadata

        Raises:
            FeatureStoreConnectionError: If connection to feature store fails
            FeatureNotFoundError: If the indicator is not found
        """
        try:
            # Check cache first if enabled
            if self.use_cache:
                cache_key = f"indicator_metadata_{indicator_id}"
                cached_data = self.cache.get(cache_key)
                if cached_data is not None:
                    return cached_data

            # Build URL
            url = f"{self.base_url}/indicators/{indicator_id}"

            # Make the request
            session = await self.get_session()
            async with session.get(url) as response:
                if response.status == 200:
                    data = await parse_json_response(response)

                    # Cache the result if caching is enabled
                    if self.use_cache:
                        self.cache.set(cache_key, data, ttl=self.cache_ttl * 2)  # Longer TTL for metadata

                    return data
                elif response.status == 404:
                    raise FeatureNotFoundError(f"Indicator '{indicator_id}' not found")
                else:
                    error_text = await response.text()
                    raise FeatureStoreConnectionError(f"Error fetching indicator metadata: {error_text}")

        except Exception as e:
            self.logger.error(f"Error retrieving indicator metadata: {str(e)}")
            raise FeatureStoreConnectionError(f"Error retrieving indicator metadata: {str(e)}")

    def _format_date(self, date: Union[str, datetime]) -> str:
        """
        Format a date as ISO string.

        Args:
            date: Date to format (string or datetime)

        Returns:
            ISO formatted date string
        """
        if isinstance(date, datetime):
            return date.isoformat()
        return date

    def _get_fallback_ohlcv_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str
    ) -> pd.DataFrame:
        """
        Generate fallback OHLCV data when the feature store is unavailable.

        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe

        Returns:
            DataFrame with synthetic OHLCV data
        """
        self.logger.warning(f"Generating fallback OHLCV data for {symbol}")

        # Convert dates to datetime if they are strings
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

        # Determine time delta based on timeframe
        if timeframe == '1m':
            delta = timedelta(minutes=1)
        elif timeframe == '5m':
            delta = timedelta(minutes=5)
        elif timeframe == '15m':
            delta = timedelta(minutes=15)
        elif timeframe == '30m':
            delta = timedelta(minutes=30)
        elif timeframe == '1h':
            delta = timedelta(hours=1)
        elif timeframe == '4h':
            delta = timedelta(hours=4)
        elif timeframe == '1d':
            delta = timedelta(days=1)
        else:
            delta = timedelta(hours=1)  # Default to 1h

        # Generate timestamps
        timestamps = []
        current = start_date
        while current <= end_date:
            timestamps.append(current)
            current += delta

        # Generate synthetic data
        base_price = 1.0 if 'USD' in symbol else 100.0

        # Add some randomness based on the symbol
        import hashlib
        symbol_hash = int(hashlib.md5(symbol.encode()).hexdigest(), 16) % 10000
        base_price += symbol_hash / 10000

        # Generate OHLCV data with some randomness
        np.random.seed(symbol_hash)

        data = {
            'timestamp': timestamps,
            'open': [base_price + np.random.normal(0, 0.01) for _ in timestamps],
            'high': [],
            'low': [],
            'close': [],
            'volume': [np.random.randint(1000, 10000) for _ in timestamps]
        }

        # Generate high, low, close based on open
        for i, open_price in enumerate(data['open']):
            close = open_price + np.random.normal(0, 0.005)
            data['close'].append(close)
            data['high'].append(max(open_price, close) + abs(np.random.normal(0, 0.003)))
            data['low'].append(min(open_price, close) - abs(np.random.normal(0, 0.003)))

        return pd.DataFrame(data)

    def _get_fallback_indicator_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate fallback indicator data when the feature store is unavailable.

        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe
            indicators: List of indicator names

        Returns:
            DataFrame with synthetic indicator data
        """
        # First, get fallback OHLCV data to base indicators on
        ohlcv_data = self._get_fallback_ohlcv_data(symbol, start_date, end_date, timeframe)

        if ohlcv_data.empty:
            return pd.DataFrame()

        # If no indicators are specified, use a default set
        if not indicators:
            indicators = ['sma_5', 'sma_20', 'ema_5', 'ema_20', 'rsi_14']

        # Create a new DataFrame with the timestamp column
        df = pd.DataFrame({'timestamp': ohlcv_data['timestamp']})

        # Generate indicator data based on OHLCV
        for indicator in indicators:
            if 'sma' in indicator:
                # Extract window from indicator name, or default to 5
                window = int(indicator.split('_')[1]) if '_' in indicator else 5
                df[indicator] = ohlcv_data['close'].rolling(window=min(window, 3)).mean()

            elif 'ema' in indicator:
                window = int(indicator.split('_')[1]) if '_' in indicator else 5
                df[indicator] = ohlcv_data['close'].ewm(span=min(window, 3)).mean()

            elif 'rsi' in indicator:
                # Simplified RSI calculation for fallback data
                df[indicator] = 50 + np.random.normal(0, 10, len(ohlcv_data))
                df[indicator] = df[indicator].clip(0, 100)  # Ensure RSI is between 0 and 100

            elif 'macd' in indicator:
                # Simplified MACD for fallback data
                df[indicator] = np.random.normal(0, 0.001, len(ohlcv_data))

            elif 'bollinger' in indicator:
                # Simplified Bollinger Bands for fallback data
                mean = ohlcv_data['close'].rolling(window=20).mean()
                df[f"{indicator}_middle"] = mean
                std = ohlcv_data['close'].rolling(window=20).std()
                df[f"{indicator}_upper"] = mean + 2 * std
                df[f"{indicator}_lower"] = mean - 2 * std

            else:
                # For any other indicator, just use random data
                df[indicator] = np.random.normal(0, 0.01, len(ohlcv_data))

        return df

    def _get_fallback_feature_data(
        self,
        feature_name: str,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        timeframe: str
    ) -> pd.DataFrame:
        """
        Generate fallback feature data when the feature store is unavailable.

        Args:
            feature_name: Name of the feature
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe

        Returns:
            DataFrame with synthetic feature data
        """
        # Get fallback OHLCV data to base feature on
        ohlcv_data = self._get_fallback_ohlcv_data(symbol, start_date, end_date, timeframe)

        if ohlcv_data.empty:
            return pd.DataFrame()

        # Create a new DataFrame with the timestamp column
        df = pd.DataFrame({'timestamp': ohlcv_data['timestamp']})

        # Generate synthetic feature data
        if 'volatility' in feature_name.lower():
            # Synthetic volatility feature
            df[feature_name] = np.abs(np.diff(ohlcv_data['close'], prepend=ohlcv_data['close'].iloc[0])) / ohlcv_data['close']
        elif 'momentum' in feature_name.lower():
            # Synthetic momentum feature
            df[feature_name] = np.random.normal(0, 0.01, len(ohlcv_data))
        elif 'regime' in feature_name.lower():
            # Synthetic market regime feature
            regimes = ['bullish', 'bearish', 'neutral', 'volatile']
            df[feature_name] = np.random.choice(regimes, len(ohlcv_data))
        else:
            # Generic synthetic feature
            df[feature_name] = np.random.normal(0, 0.01, len(ohlcv_data))

        return df
