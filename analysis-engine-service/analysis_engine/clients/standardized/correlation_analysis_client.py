"""
Standardized Correlation Analysis Client

This module provides a client for interacting with the standardized Correlation Analysis API.
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

class CorrelationAnalysisClient:
    """
    Client for interacting with the standardized Correlation Analysis API.
    
    This client provides methods for analyzing correlations between currency pairs,
    lead-lag relationships, correlation breakdowns, and cointegration.
    
    It includes resilience patterns like retry with backoff and circuit breaking.
    """

    def __init__(self, base_url: Optional[str]=None, timeout: int=30):
        """
        Initialize the Correlation Analysis client.
        
        Args:
            base_url: Base URL for the Correlation Analysis API. If None, uses the URL from settings.
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.base_url = base_url or settings.analysis_engine_url
        self.timeout = timeout
        self.api_prefix = '/api/v1/analysis/correlations'
        self.circuit_breaker = circuit_breaker(failure_threshold=5,
            recovery_timeout=30, name='correlation_analysis_client')
        logger.info(
            f'Initialized Correlation Analysis client with base URL: {self.base_url}'
            )

    @async_with_exception_handling
    async def _make_request(self, method: str, endpoint: str, data:
        Optional[Dict]=None, params: Optional[Dict]=None) ->Dict:
        """
        Make a request to the Correlation Analysis API with resilience patterns.
        
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
                                f'Server error from Correlation Analysis API: {error_text}'
                                )
                            raise ServiceUnavailableError(
                                f'Correlation Analysis API server error: {response.status}'
                                )
                        if response.status >= 400:
                            error_text = await response.text()
                            logger.error(
                                f'Client error from Correlation Analysis API: {error_text}'
                                )
                            raise Exception(
                                f'Correlation Analysis API client error: {response.status} - {error_text}'
                                )
                        return await response.json()
            except aiohttp.ClientError as e:
                logger.error(
                    f'Connection error to Correlation Analysis API: {str(e)}')
                raise ServiceUnavailableError(
                    f'Failed to connect to Correlation Analysis API: {str(e)}')
            except asyncio.TimeoutError:
                logger.error(f'Timeout connecting to Correlation Analysis API')
                raise ServiceTimeoutError(
                    f'Timeout connecting to Correlation Analysis API')
        return await _request()

    @with_analysis_resilience('analyze_currency_correlations')
    async def analyze_currency_correlations(self, data: Dict[str, Dict[str,
        Any]], window_sizes: Optional[List[int]]=None, correlation_method:
        Optional[str]='pearson', significance_threshold: Optional[float]=0.7
        ) ->Dict:
        """
        Analyze correlations between currency pairs with enhanced features.
        
        Args:
            data: Currency pair data for correlation analysis
            window_sizes: Correlation window sizes in days
            correlation_method: Correlation method (pearson or spearman)
            significance_threshold: Threshold for significant correlation
            
        Returns:
            Correlation analysis results
        """
        request_data = {'data': data}
        params = {}
        if window_sizes:
            params['window_sizes'] = window_sizes
        if correlation_method:
            params['correlation_method'] = correlation_method
        if significance_threshold:
            params['significance_threshold'] = significance_threshold
        logger.info(f'Analyzing correlations for {len(data)} currency pairs')
        return await self._make_request('POST', '/analyze', data=
            request_data, params=params)

    @with_analysis_resilience('analyze_lead_lag_relationships')
    async def analyze_lead_lag_relationships(self, data: Dict[str, Dict[str,
        Any]], max_lag: Optional[int]=10, significance: Optional[float]=0.05
        ) ->Dict:
        """
        Analyze lead-lag relationships between currency pairs.
        
        Args:
            data: Currency pair data for lead-lag analysis
            max_lag: Maximum lag for Granger causality test
            significance: P-value threshold for significance
            
        Returns:
            Lead-lag analysis results
        """
        request_data = {'data': data}
        params = {}
        if max_lag:
            params['max_lag'] = max_lag
        if significance:
            params['significance'] = significance
        logger.info(
            f'Analyzing lead-lag relationships for {len(data)} currency pairs')
        return await self._make_request('POST', '/lead-lag', data=
            request_data, params=params)

    async def detect_correlation_breakdowns(self, data: Dict[str, Dict[str,
        Any]], short_window: Optional[int]=5, long_window: Optional[int]=60,
        change_threshold: Optional[float]=0.3) ->Dict:
        """
        Detect significant breakdowns in correlation patterns between currency pairs.
        
        Args:
            data: Currency pair data for correlation breakdown detection
            short_window: Short-term correlation window
            long_window: Long-term correlation window for comparison
            change_threshold: Threshold for significant correlation change
            
        Returns:
            Correlation breakdown detection results
        """
        request_data = {'data': data}
        params = {}
        if short_window:
            params['short_window'] = short_window
        if long_window:
            params['long_window'] = long_window
        if change_threshold:
            params['change_threshold'] = change_threshold
        logger.info(
            f'Detecting correlation breakdowns for {len(data)} currency pairs')
        return await self._make_request('POST', '/breakdown-detection',
            data=request_data, params=params)

    async def test_pair_cointegration(self, data: Dict[str, Dict[str, Any]],
        significance: Optional[float]=0.05) ->Dict:
        """
        Test for cointegration between currency pairs.
        
        Args:
            data: Currency pair data for cointegration testing
            significance: P-value threshold for cointegration significance
            
        Returns:
            Cointegration test results
        """
        request_data = {'data': data}
        params = {}
        if significance:
            params['significance'] = significance
        logger.info(f'Testing cointegration for {len(data)} currency pairs')
        return await self._make_request('POST', '/cointegration', data=
            request_data, params=params)
