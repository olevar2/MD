"""
Standardized Manipulation Detection Client

This module provides a client for interacting with the standardized Manipulation Detection API.
"""
import logging
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from analysis_engine.core.config import get_settings
from analysis_engine.core.resilience import retry_with_backoff, circuit_breaker
from analysis_engine.monitoring.structured_logging import get_structured_logger
from analysis_engine.core.exceptions_bridge import ServiceUnavailableError, ServiceTimeoutError
logger = get_structured_logger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ManipulationDetectionClient:
    """
    Client for interacting with the standardized Manipulation Detection API.
    
    This client provides methods for detecting potential market manipulation
    patterns in forex data, including stop hunting, fake breakouts, and unusual
    price-volume relationships.
    
    It includes resilience patterns like retry with backoff and circuit breaking.
    """

    def __init__(self, base_url: Optional[str]=None, timeout: int=30):
        """
        Initialize the Manipulation Detection client.
        
        Args:
            base_url: Base URL for the Manipulation Detection API. If None, uses the URL from settings.
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.base_url = base_url or settings.analysis_engine_url
        self.timeout = timeout
        self.api_prefix = '/api/v1/analysis/manipulation-detection'
        self.circuit_breaker = circuit_breaker(failure_threshold=5,
            recovery_timeout=30, name='manipulation_detection_client')
        logger.info(
            f'Initialized Manipulation Detection client with base URL: {self.base_url}'
            )

    @async_with_exception_handling
    async def _make_request(self, method: str, endpoint: str, data:
        Optional[Dict]=None, params: Optional[Dict]=None) ->Dict:
        """
        Make a request to the Manipulation Detection API with resilience patterns.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data
            params: Query parameters
            
        Returns:
            Response data
            
        Raises:
            ServiceUnavailableError: If the service is unavailable
            ServiceTimeoutError: If the request times out
            Exception: For other errors
        """
        url = f'{self.base_url}{self.api_prefix}{endpoint}'

        @retry_with_backoff(max_retries=3, backoff_factor=1.5,
            retry_exceptions=[aiohttp.ClientError, TimeoutError])
        @self.circuit_breaker
        @async_with_exception_handling
        async def _request():
    """
     request.
    
    """

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(method=method, url=url, json
                        =data, params=params, timeout=self.timeout
                        ) as response:
                        if response.status >= 500:
                            error_text = await response.text()
                            logger.error(
                                f'Server error from Manipulation Detection API: {error_text}'
                                )
                            raise ServiceUnavailableError(
                                f'Manipulation Detection API server error: {response.status}'
                                )
                        if response.status >= 400:
                            error_text = await response.text()
                            logger.error(
                                f'Client error from Manipulation Detection API: {error_text}'
                                )
                            raise Exception(
                                f'Manipulation Detection API client error: {response.status} - {error_text}'
                                )
                        return await response.json()
            except aiohttp.ClientError as e:
                logger.error(
                    f'Connection error to Manipulation Detection API: {str(e)}'
                    )
                raise ServiceUnavailableError(
                    f'Failed to connect to Manipulation Detection API: {str(e)}'
                    )
            except asyncio.TimeoutError:
                logger.error(
                    f'Timeout connecting to Manipulation Detection API')
                raise ServiceTimeoutError(
                    f'Timeout connecting to Manipulation Detection API')
        return await _request()

    async def detect_manipulation_patterns(self, ohlcv: List[Dict[str, Any]
        ], metadata: Dict[str, Any], sensitivity: Optional[float]=1.0,
        include_protection: Optional[bool]=True) ->Dict:
        """
        Analyze market data for potential manipulation patterns.
        
        Args:
            ohlcv: OHLCV data points
            metadata: Market data metadata
            sensitivity: Detection sensitivity multiplier (0.5-2.0)
            include_protection: Include protection recommendations
            
        Returns:
            Manipulation detection results
        """
        data = {'ohlcv': ohlcv, 'metadata': metadata}
        params = {}
        if sensitivity != 1.0:
            params['sensitivity'] = sensitivity
        if not include_protection:
            params['include_protection'] = include_protection
        logger.info(
            f"Detecting manipulation patterns for {metadata.get('symbol')}")
        return await self._make_request('POST', '/detect', data=data,
            params=params)

    async def detect_stop_hunting(self, ohlcv: List[Dict[str, Any]],
        metadata: Dict[str, Any], lookback: Optional[int]=30,
        recovery_threshold: Optional[float]=0.5) ->Dict:
        """
        Specifically analyze for stop hunting patterns.
        
        Args:
            ohlcv: OHLCV data points
            metadata: Market data metadata
            lookback: Lookback period for stop hunting detection
            recovery_threshold: Recovery percentage threshold
            
        Returns:
            Stop hunting detection results
        """
        data = {'ohlcv': ohlcv, 'metadata': metadata}
        params = {}
        if lookback != 30:
            params['lookback'] = lookback
        if recovery_threshold != 0.5:
            params['recovery_threshold'] = recovery_threshold
        logger.info(
            f"Detecting stop hunting patterns for {metadata.get('symbol')}")
        return await self._make_request('POST', '/stop-hunting', data=data,
            params=params)

    async def detect_fake_breakouts(self, ohlcv: List[Dict[str, Any]],
        metadata: Dict[str, Any], threshold: Optional[float]=0.7) ->Dict:
        """
        Specifically analyze for fake breakout patterns.
        
        Args:
            ohlcv: OHLCV data points
            metadata: Market data metadata
            threshold: Fake breakout detection threshold
            
        Returns:
            Fake breakout detection results
        """
        data = {'ohlcv': ohlcv, 'metadata': metadata}
        params = {}
        if threshold != 0.7:
            params['threshold'] = threshold
        logger.info(
            f"Detecting fake breakout patterns for {metadata.get('symbol')}")
        return await self._make_request('POST', '/fake-breakouts', data=
            data, params=params)

    async def detect_volume_anomalies(self, ohlcv: List[Dict[str, Any]],
        metadata: Dict[str, Any], z_threshold: Optional[float]=2.0) ->Dict:
        """
        Specifically analyze for volume anomalies.
        
        Args:
            ohlcv: OHLCV data points
            metadata: Market data metadata
            z_threshold: Z-score threshold for volume anomaly detection
            
        Returns:
            Volume anomaly detection results
        """
        data = {'ohlcv': ohlcv, 'metadata': metadata}
        params = {}
        if z_threshold != 2.0:
            params['z_threshold'] = z_threshold
        logger.info(f"Detecting volume anomalies for {metadata.get('symbol')}")
        return await self._make_request('POST', '/volume-anomalies', data=
            data, params=params)
