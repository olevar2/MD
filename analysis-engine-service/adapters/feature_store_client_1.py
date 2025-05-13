"""
Feature Store Client Module

This module provides a client for interacting with the feature store service
to retrieve pre-calculated indicators and market data.

This implementation includes resilience patterns:
1. Circuit breakers to prevent cascading failures
2. Retry mechanisms with backoff for transient failures
3. Bulkheads to isolate critical operations
4. Timeout handling for all external operations
"""
import logging
import requests
import json
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from datetime import datetime, timedelta
from analysis_engine.config.settings import Settings
from analysis_engine.utils.exceptions import FeatureStoreError
from analysis_engine.resilience import create_circuit_breaker, CircuitBreakerConfig, CircuitBreakerOpen, retry_with_policy, timeout_handler, bulkhead
from analysis_engine.resilience.utils import with_resilience
from analysis_engine.resilience.config import get_circuit_breaker_config, get_retry_config, get_timeout_config, get_bulkhead_config
logger = logging.getLogger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class FeatureStoreClient:
    """
    Client for the feature store service.

    This class provides methods to fetch indicators, market data,
    and request on-demand recalculation of indicators.

    Includes resilience patterns:
    - Circuit breakers for preventing cascading failures
    - Retry mechanisms for transient failures
    - Bulkheads for isolating critical operations
    - Timeout handling for all external operations
    """

    def __init__(self, settings: Settings=None):
        """
        Initialize the feature store client.

        Args:
            settings: Application settings containing feature store configuration
        """
        self.settings = settings or Settings()
        self.base_url = self.settings.feature_store_service.base_url
        self.timeout = self.settings.feature_store_service.timeout
        self.logger = logging.getLogger(__name__)
        self.indicator_cb = create_circuit_breaker(service_name=
            'analysis_engine', resource_name='feature_store_indicator',
            config=get_circuit_breaker_config('feature_store'))
        self.market_data_cb = create_circuit_breaker(service_name=
            'analysis_engine', resource_name='feature_store_market_data',
            config=get_circuit_breaker_config('feature_store'))
        self.recalculate_cb = create_circuit_breaker(service_name=
            'analysis_engine', resource_name='feature_store_recalculate',
            config=get_circuit_breaker_config('feature_store'))
        self.metadata_cb = create_circuit_breaker(service_name=
            'analysis_engine', resource_name='feature_store_metadata',
            config=get_circuit_breaker_config('feature_store'))

    @with_resilience(service_name='analysis_engine', operation_name=
        'get_indicator', service_type='feature_store', exceptions=[requests
        .RequestException, requests.Timeout, requests.ConnectionError])
    @with_exception_handling
    def get_indicator(self, indicator_type: str, symbol: str, params: Dict[
        str, Any]=None, start_date: Optional[Union[str, datetime]]=None,
        end_date: Optional[Union[str, datetime]]=None, timeframe: str='1d',
        cache_only: bool=True) ->pd.DataFrame:
        """
        Fetch indicator data from the feature store.

        Args:
            indicator_type: Type of indicator to fetch (e.g., 'rsi', 'ema', 'macd')
            symbol: Trading symbol/instrument
            params: Parameters for the indicator
            start_date: Start date for the data (optional)
            end_date: End date for the data (optional)
            timeframe: Data timeframe (e.g., '1m', '1h', '1d')
            cache_only: If True, only fetch cached data; if False, calculate on-demand

        Returns:
            DataFrame containing the indicator data

        Raises:
            FeatureStoreError: If there's an error fetching the data
        """
        try:
            params = params or {}
            if isinstance(start_date, datetime):
                start_date = start_date.isoformat()
            if isinstance(end_date, datetime):
                end_date = end_date.isoformat()
            endpoint = f'/api/v1/indicators/{indicator_type}'
            request_data = {'symbol': symbol, 'parameters': params,
                'timeframe': timeframe, 'cache_only': cache_only}
            if start_date:
                request_data['start_date'] = start_date
            if end_date:
                request_data['end_date'] = end_date
            response = requests.post(f'{self.base_url}{endpoint}', json=
                request_data, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    return pd.DataFrame(data['data'])
                else:
                    self.logger.warning(
                        f'No indicator data returned for {indicator_type} on {symbol}'
                        )
                    return pd.DataFrame()
            else:
                error_msg = (
                    f'Error fetching {indicator_type} indicator: {response.text}'
                    )
                self.logger.error(error_msg)
                raise FeatureStoreError(error_msg)
        except requests.RequestException as e:
            error_msg = (
                f'Request error fetching {indicator_type} indicator: {str(e)}')
            self.logger.error(error_msg)
            raise FeatureStoreError(error_msg)
        except Exception as e:
            error_msg = (
                f'Unexpected error fetching {indicator_type} indicator: {str(e)}'
                )
            self.logger.error(error_msg)
            raise FeatureStoreError(error_msg)

    @with_resilience(service_name='analysis_engine', operation_name=
        'get_market_data', service_type='feature_store', exceptions=[
        requests.RequestException, requests.Timeout, requests.ConnectionError])
    @with_exception_handling
    def get_market_data(self, symbol: str, start_date: Optional[Union[str,
        datetime]]=None, end_date: Optional[Union[str, datetime]]=None,
        timeframe: str='1d', lookback: int=None) ->pd.DataFrame:
        """
        Fetch market data from the feature store.

        Args:
            symbol: Trading symbol/instrument
            start_date: Start date for the data
            end_date: End date for the data
            timeframe: Data timeframe (e.g., '1m', '1h', '1d')
            lookback: Number of periods to fetch (used instead of start_date)

        Returns:
            DataFrame containing market data (OHLCV)

        Raises:
            FeatureStoreError: If there's an error fetching the data
        """
        try:
            if isinstance(start_date, datetime):
                start_date = start_date.isoformat()
            if isinstance(end_date, datetime):
                end_date = end_date.isoformat()
            if lookback and not start_date:
                if not end_date:
                    end_date = datetime.now()
                if isinstance(end_date, str):
                    end_dt = pd.to_datetime(end_date)
                else:
                    end_dt = end_date
                if timeframe.endswith('m'):
                    minutes = int(timeframe[:-1])
                    start_date = (end_dt - timedelta(minutes=minutes *
                        lookback)).isoformat()
                elif timeframe.endswith('h'):
                    hours = int(timeframe[:-1])
                    start_date = (end_dt - timedelta(hours=hours * lookback)
                        ).isoformat()
                elif timeframe.endswith('d'):
                    days = int(timeframe[:-1])
                    start_date = (end_dt - timedelta(days=days * lookback)
                        ).isoformat()
                else:
                    start_date = (end_dt - timedelta(days=lookback)).isoformat(
                        )
            endpoint = '/api/v1/market-data'
            request_data = {'symbol': symbol, 'timeframe': timeframe}
            if start_date:
                request_data['start_date'] = start_date
            if end_date:
                request_data['end_date'] = end_date
            response = requests.post(f'{self.base_url}{endpoint}', json=
                request_data, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    df = pd.DataFrame(data['data'])
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    return df
                else:
                    self.logger.warning(f'No market data returned for {symbol}'
                        )
                    return pd.DataFrame()
            else:
                error_msg = f'Error fetching market data: {response.text}'
                self.logger.error(error_msg)
                raise FeatureStoreError(error_msg)
        except requests.RequestException as e:
            error_msg = f'Request error fetching market data: {str(e)}'
            self.logger.error(error_msg)
            raise FeatureStoreError(error_msg)
        except Exception as e:
            error_msg = f'Unexpected error fetching market data: {str(e)}'
            self.logger.error(error_msg)
            raise FeatureStoreError(error_msg)

    @with_resilience(service_name='analysis_engine', operation_name=
        'recalculate_indicator', service_type='feature_store', exceptions=[
        requests.RequestException, requests.Timeout, requests.ConnectionError])
    @with_exception_handling
    def recalculate_indicator(self, indicator_type: str, symbol: str,
        params: Dict[str, Any]=None, timeframes: List[str]=None, priority:
        str='normal') ->Dict[str, Any]:
        """
        Request on-demand recalculation of an indicator.

        Args:
            indicator_type: Type of indicator to recalculate
            symbol: Trading symbol/instrument
            params: Parameters for the indicator
            timeframes: List of timeframes to recalculate (e.g., ['1m', '1h', '1d'])
            priority: Priority of the calculation request ('low', 'normal', 'high')

        Returns:
            Dictionary with job information

        Raises:
            FeatureStoreError: If there's an error requesting the recalculation
        """
        try:
            params = params or {}
            timeframes = timeframes or ['1d']
            endpoint = f'/api/v1/indicators/{indicator_type}/recalculate'
            request_data = {'symbol': symbol, 'parameters': params,
                'timeframes': timeframes, 'priority': priority}
            response = requests.post(f'{self.base_url}{endpoint}', json=
                request_data, timeout=self.timeout)
            if response.status_code in (200, 202):
                return response.json()
            else:
                error_msg = (
                    f'Error requesting indicator recalculation: {response.text}'
                    )
                self.logger.error(error_msg)
                raise FeatureStoreError(error_msg)
        except requests.RequestException as e:
            error_msg = f'Request error for indicator recalculation: {str(e)}'
            self.logger.error(error_msg)
            raise FeatureStoreError(error_msg)
        except Exception as e:
            error_msg = (
                f'Unexpected error for indicator recalculation: {str(e)}')
            self.logger.error(error_msg)
            raise FeatureStoreError(error_msg)

    @with_resilience(service_name='analysis_engine', operation_name=
        'get_available_indicators', service_type='feature_store',
        exceptions=[requests.RequestException, requests.Timeout, requests.
        ConnectionError])
    @with_exception_handling
    def get_available_indicators(self) ->List[Dict[str, Any]]:
        """
        Get a list of available indicators in the feature store.

        Returns:
            List of dictionaries with indicator metadata

        Raises:
            FeatureStoreError: If there's an error fetching the indicator list
        """
        try:
            endpoint = '/api/v1/indicators'
            response = requests.get(f'{self.base_url}{endpoint}', timeout=
                self.timeout)
            if response.status_code == 200:
                data = response.json()
                return data.get('indicators', [])
            else:
                error_msg = (
                    f'Error fetching available indicators: {response.text}')
                self.logger.error(error_msg)
                raise FeatureStoreError(error_msg)
        except requests.RequestException as e:
            error_msg = (
                f'Request error fetching available indicators: {str(e)}')
            self.logger.error(error_msg)
            raise FeatureStoreError(error_msg)
        except Exception as e:
            error_msg = (
                f'Unexpected error fetching available indicators: {str(e)}')
            self.logger.error(error_msg)
            raise FeatureStoreError(error_msg)
