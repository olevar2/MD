"""
Standardized Tool Effectiveness Client

This module provides a client for interacting with the standardized Tool Effectiveness API.
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

class EffectivenessClient:
    """
    Client for interacting with the standardized Tool Effectiveness API.
    
    This client provides methods for tracking and analyzing the effectiveness
    of trading tools, including signal registration, outcome tracking, and
    performance metrics.
    
    It includes resilience patterns like retry with backoff and circuit breaking.
    """

    def __init__(self, base_url: Optional[str]=None, timeout: int=30):
        """
        Initialize the Tool Effectiveness client.
        
        Args:
            base_url: Base URL for the Tool Effectiveness API. If None, uses the URL from settings.
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.base_url = base_url or settings.analysis_engine_url
        self.timeout = timeout
        self.api_prefix = '/api/v1/analysis/effectiveness'
        self.circuit_breaker = circuit_breaker(failure_threshold=5,
            recovery_timeout=30, name='effectiveness_client')
        logger.info(
            f'Initialized Tool Effectiveness client with base URL: {self.base_url}'
            )

    @async_with_exception_handling
    async def _make_request(self, method: str, endpoint: str, data:
        Optional[Dict]=None, params: Optional[Dict]=None) ->Dict:
        """
        Make a request to the Tool Effectiveness API with resilience patterns.
        
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
                                f'Server error from Tool Effectiveness API: {error_text}'
                                )
                            raise ServiceUnavailableError(
                                f'Tool Effectiveness API server error: {response.status}'
                                )
                        if response.status >= 400:
                            error_text = await response.text()
                            logger.error(
                                f'Client error from Tool Effectiveness API: {error_text}'
                                )
                            raise Exception(
                                f'Tool Effectiveness API client error: {response.status} - {error_text}'
                                )
                        return await response.json()
            except aiohttp.ClientError as e:
                logger.error(
                    f'Connection error to Tool Effectiveness API: {str(e)}')
                raise ServiceUnavailableError(
                    f'Failed to connect to Tool Effectiveness API: {str(e)}')
            except asyncio.TimeoutError:
                logger.error(f'Timeout connecting to Tool Effectiveness API')
                raise ServiceTimeoutError(
                    f'Timeout connecting to Tool Effectiveness API')
        return await _request()

    async def register_signal(self, tool_id: str, signal_type: str,
        instrument: str, timestamp: Union[datetime, str], confidence: float
        =1.0, timeframe: str='1H', market_regime: str='unknown',
        additional_data: Optional[Dict]=None) ->Dict:
        """
        Register a new signal from a trading tool for effectiveness tracking.
        
        Args:
            tool_id: Identifier for the trading tool that generated the signal
            signal_type: Type of signal (buy, sell, etc.)
            instrument: Trading instrument (e.g., 'EUR_USD')
            timestamp: When the signal was generated
            confidence: Signal confidence level (0.0-1.0)
            timeframe: Timeframe of the analysis (e.g., '1H', '4H', '1D')
            market_regime: Market regime at signal time
            additional_data: Any additional signal metadata
            
        Returns:
            Signal registration response
        """
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        data = {'tool_id': tool_id, 'signal_type': signal_type,
            'instrument': instrument, 'timestamp': timestamp, 'confidence':
            confidence, 'timeframe': timeframe, 'market_regime':
            market_regime, 'additional_data': additional_data or {}}
        logger.info(f'Registering signal for tool {tool_id}')
        return await self._make_request('POST', '/signals', data=data)

    async def register_outcome(self, signal_id: str, success: bool,
        realized_profit: float=0.0, timestamp: Optional[Union[datetime, str
        ]]=None, additional_data: Optional[Dict]=None) ->Dict:
        """
        Register the outcome of a previously registered signal.
        
        Args:
            signal_id: ID of the signal that this outcome is associated with
            success: Whether the signal led to a successful trade
            realized_profit: Profit/loss realized from the trade
            timestamp: When the outcome was recorded
            additional_data: Any additional outcome metadata
            
        Returns:
            Outcome registration response
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()
        elif isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        data = {'signal_id': signal_id, 'success': success,
            'realized_profit': realized_profit, 'timestamp': timestamp,
            'additional_data': additional_data or {}}
        logger.info(f'Registering outcome for signal {signal_id}')
        return await self._make_request('POST', '/outcomes', data=data)

    @with_resilience('get_effectiveness_metrics')
    async def get_effectiveness_metrics(self, tool_id: Optional[str]=None,
        timeframe: Optional[str]=None, instrument: Optional[str]=None,
        market_regime: Optional[str]=None, from_date: Optional[Union[
        datetime, str]]=None, to_date: Optional[Union[datetime, str]]=None
        ) ->List[Dict]:
        """
        Retrieve effectiveness metrics for trading tools with optional filtering.
        
        Args:
            tool_id: Filter by specific tool ID
            timeframe: Filter by specific timeframe
            instrument: Filter by specific instrument
            market_regime: Filter by market regime
            from_date: Start date for metrics calculation
            to_date: End date for metrics calculation
            
        Returns:
            List of tool effectiveness metrics
        """
        params = {}
        if tool_id:
            params['tool_id'] = tool_id
        if timeframe:
            params['timeframe'] = timeframe
        if instrument:
            params['instrument'] = instrument
        if market_regime:
            params['market_regime'] = market_regime
        if from_date:
            params['from_date'] = from_date.isoformat() if isinstance(from_date
                , datetime) else from_date
        if to_date:
            params['to_date'] = to_date.isoformat() if isinstance(to_date,
                datetime) else to_date
        logger.info(
            f"Getting effectiveness metrics for tool {tool_id if tool_id else 'all tools'}"
            )
        return await self._make_request('GET', '/metrics', params=params)

    @with_resilience('get_dashboard_data')
    async def get_dashboard_data(self) ->Dict:
        """
        Get aggregated data suitable for dashboard visualization.
        
        Returns:
            Dashboard data
        """
        logger.info('Getting dashboard data')
        return await self._make_request('GET', '/dashboard-data')

    async def save_effectiveness_report(self, name: str, description:
        Optional[str]=None, tool_id: Optional[str]=None, timeframe:
        Optional[str]=None, instrument: Optional[str]=None, market_regime:
        Optional[str]=None, from_date: Optional[Union[datetime, str]]=None,
        to_date: Optional[Union[datetime, str]]=None) ->Dict:
        """
        Save a new effectiveness report.
        
        Args:
            name: Name of the report
            description: Description of the report
            tool_id: Filter by specific tool ID
            timeframe: Filter by specific timeframe
            instrument: Filter by specific instrument
            market_regime: Filter by market regime
            from_date: Start date for metrics
            to_date: End date for metrics
            
        Returns:
            Report creation response
        """
        data = {'name': name, 'description': description}
        if tool_id:
            data['tool_id'] = tool_id
        if timeframe:
            data['timeframe'] = timeframe
        if instrument:
            data['instrument'] = instrument
        if market_regime:
            data['market_regime'] = market_regime
        if from_date:
            data['from_date'] = from_date.isoformat() if isinstance(from_date,
                datetime) else from_date
        if to_date:
            data['to_date'] = to_date.isoformat() if isinstance(to_date,
                datetime) else to_date
        logger.info(f'Saving effectiveness report: {name}')
        return await self._make_request('POST', '/reports', data=data)

    @with_resilience('get_effectiveness_reports')
    async def get_effectiveness_reports(self, skip: int=0, limit: int=100
        ) ->List[Dict]:
        """
        Get all saved effectiveness reports.
        
        Args:
            skip: Skip items for pagination
            limit: Limit items for pagination
            
        Returns:
            List of report summaries
        """
        params = {'skip': skip, 'limit': limit}
        logger.info('Getting effectiveness reports')
        return await self._make_request('GET', '/reports', params=params)

    @with_resilience('get_effectiveness_report')
    async def get_effectiveness_report(self, report_id: int) ->Dict:
        """
        Get a specific effectiveness report by ID.
        
        Args:
            report_id: ID of the report to retrieve
            
        Returns:
            Report details
        """
        logger.info(f'Getting effectiveness report {report_id}')
        return await self._make_request('GET', f'/reports/{report_id}')

    async def clear_tool_data(self, tool_id: str) ->Dict:
        """
        Clear all data for a specific tool (for testing or resetting purposes).
        
        Args:
            tool_id: ID of the tool to clear data for
            
        Returns:
            Deletion response
        """
        logger.info(f'Clearing data for tool {tool_id}')
        return await self._make_request('DELETE', f'/tools/{tool_id}/data')
