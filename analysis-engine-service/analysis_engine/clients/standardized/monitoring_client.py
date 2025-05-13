"""
Standardized Monitoring Client

This module provides a client for interacting with the standardized Monitoring API.
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
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class MonitoringClient:
    """
    Client for interacting with the standardized Monitoring API.
    
    This client provides methods for accessing monitoring-related functionality,
    including async performance metrics, memory usage, and health status.
    
    It includes resilience patterns like retry with backoff and circuit breaking.
    """

    def __init__(self, base_url: Optional[str]=None, timeout: int=30):
        """
        Initialize the Monitoring client.
        
        Args:
            base_url: Base URL for the Monitoring API. If None, uses the URL from settings.
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.base_url = base_url or settings.analysis_engine_url
        self.timeout = timeout
        self.api_prefix = '/api/v1/analysis/monitoring'
        self.circuit_breaker = circuit_breaker(failure_threshold=5,
            recovery_timeout=30, name='monitoring_client')
        logger.info(
            f'Initialized Monitoring client with base URL: {self.base_url}')

    @async_with_exception_handling
    async def _make_request(self, method: str, endpoint: str, data:
        Optional[Dict]=None, params: Optional[Dict]=None) ->Dict:
        """
        Make a request to the Monitoring API with resilience patterns.
        
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
                                f'Server error from Monitoring API: {error_text}'
                                )
                            raise ServiceUnavailableError(
                                f'Monitoring API server error: {response.status}'
                                )
                        if response.status >= 400:
                            error_text = await response.text()
                            logger.error(
                                f'Client error from Monitoring API: {error_text}'
                                )
                            raise Exception(
                                f'Monitoring API client error: {response.status} - {error_text}'
                                )
                        return await response.json()
            except aiohttp.ClientError as e:
                logger.error(f'Connection error to Monitoring API: {str(e)}')
                raise ServiceUnavailableError(
                    f'Failed to connect to Monitoring API: {str(e)}')
            except asyncio.TimeoutError:
                logger.error(f'Timeout connecting to Monitoring API')
                raise ServiceTimeoutError(
                    f'Timeout connecting to Monitoring API')
        return await _request()

    @with_resilience('get_async_performance_metrics')
    async def get_async_performance_metrics(self, operation: Optional[str]=None
        ) ->Dict:
        """
        Get async performance metrics.
        
        Args:
            operation: Optional operation name to filter by
            
        Returns:
            Async performance metrics
        """
        params = {}
        if operation:
            params['operation'] = operation
        logger.info(
            f"Getting async performance metrics for {operation if operation else 'all operations'}"
            )
        return await self._make_request('GET', '/async-performance', params
            =params)

    @with_resilience('get_memory_metrics')
    async def get_memory_metrics(self) ->Dict:
        """
        Get memory usage metrics.
        
        Returns:
            Memory usage metrics
        """
        logger.info('Getting memory metrics')
        return await self._make_request('GET', '/memory')

    async def trigger_async_performance_report(self) ->Dict:
        """
        Trigger an immediate async performance report.
        
        Returns:
            Report trigger status
        """
        logger.info('Triggering async performance report')
        return await self._make_request('POST', '/async-performance/report')

    @with_resilience('get_service_health')
    async def get_service_health(self) ->Dict:
        """
        Get detailed health status of the service and its dependencies.
        
        Returns:
            Service health status
        """
        logger.info('Getting service health')
        return await self._make_request('GET', '/health')
