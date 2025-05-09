"""
Standardized Market Regime Client

This module provides a client for interacting with the standardized Market Regime API.
"""

import logging
import aiohttp
import asyncio
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from analysis_engine.core.config import get_settings
from analysis_engine.core.resilience import retry_with_backoff, circuit_breaker
from analysis_engine.monitoring.structured_logging import get_structured_logger
from analysis_engine.core.exceptions_bridge import ServiceUnavailableError, ServiceTimeoutError

logger = get_structured_logger(__name__)

class MarketRegimeClient:
    """
    Client for interacting with the standardized Market Regime API.

    This client provides methods for detecting market regimes,
    analyzing tool effectiveness across different market conditions,
    and more.

    It includes resilience patterns like retry with backoff and circuit breaking.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: int = 30):
        """
        Initialize the Market Regime client.

        Args:
            base_url: Base URL for the Market Regime API. If None, uses the URL from settings.
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.base_url = base_url or settings.analysis_engine_url
        self.timeout = timeout
        self.api_prefix = "/api/v1/analysis/market-regimes"

        # Configure circuit breaker
        self.circuit_breaker = circuit_breaker(
            failure_threshold=5,
            recovery_timeout=30,
            name="market_regime_client"
        )

        logger.info(f"Initialized Market Regime client with base URL: {self.base_url}")

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Make a request to the Market Regime API with resilience patterns.

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
        url = f"{self.base_url}{self.api_prefix}{endpoint}"

        @retry_with_backoff(
            max_retries=3,
            backoff_factor=1.5,
            retry_exceptions=[aiohttp.ClientError, TimeoutError]
        )
        @self.circuit_breaker
        async def _request():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method=method,
                        url=url,
                        json=data,
                        params=params,
                        timeout=self.timeout
                    ) as response:
                        if response.status >= 500:
                            error_text = await response.text()
                            logger.error(f"Server error from Market Regime API: {error_text}")
                            raise ServiceUnavailableError(f"Market Regime API server error: {response.status}")

                        if response.status >= 400:
                            error_text = await response.text()
                            logger.error(f"Client error from Market Regime API: {error_text}")
                            raise Exception(f"Market Regime API client error: {response.status} - {error_text}")

                        return await response.json()
            except aiohttp.ClientError as e:
                logger.error(f"Connection error to Market Regime API: {str(e)}")
                raise ServiceUnavailableError(f"Failed to connect to Market Regime API: {str(e)}")
            except asyncio.TimeoutError:
                logger.error(f"Timeout connecting to Market Regime API")
                raise ServiceTimeoutError(f"Timeout connecting to Market Regime API")

        return await _request()

    async def detect_market_regime(
        self,
        symbol: str,
        timeframe: str,
        ohlc_data: Union[List[Dict], pd.DataFrame]
    ) -> Dict:
        """
        Detect the current market regime based on price data.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe for analysis (e.g., '1h', '4h', 'D')
            ohlc_data: OHLC price data as list of dicts or DataFrame

        Returns:
            Detected market regime
        """
        # Convert DataFrame to list of dicts if needed
        if isinstance(ohlc_data, pd.DataFrame):
            if 'timestamp' not in ohlc_data.columns and ohlc_data.index.name != 'timestamp':
                ohlc_data = ohlc_data.reset_index()

            ohlc_data = ohlc_data.to_dict('records')

        data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "ohlc_data": ohlc_data
        }

        logger.info(f"Detecting market regime for symbol {symbol}, timeframe {timeframe}")
        return await self._make_request("POST", "/detect", data=data)

    async def get_regime_history(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get historical regime data for a specific symbol and timeframe.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe for analysis (e.g., '1h', '4h', 'D')
            limit: Maximum number of history entries to return

        Returns:
            List of historical regime data
        """
        params = {
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit
        }

        logger.info(f"Getting regime history for symbol {symbol}, timeframe {timeframe}")
        return await self._make_request("GET", "/history", params=params)

    async def analyze_tool_regime_performance(
        self,
        tool_id: str,
        timeframe: Optional[str] = None,
        instrument: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Dict:
        """
        Get the performance metrics of a tool across different market regimes.

        Args:
            tool_id: Identifier for the trading tool
            timeframe: Timeframe for analysis (e.g., '1h', '4h', 'D')
            instrument: Trading instrument (e.g., 'EUR_USD')
            from_date: Start date for analysis
            to_date: End date for analysis

        Returns:
            Performance metrics across different market regimes
        """
        params = {
            "tool_id": tool_id,
            "timeframe": timeframe,
            "instrument": instrument
        }

        if from_date:
            params["from_date"] = from_date.isoformat()

        if to_date:
            params["to_date"] = to_date.isoformat()

        logger.info(f"Analyzing tool regime performance for tool {tool_id}")
        return await self._make_request("GET", "/tools/regime-analysis", params=params)

    async def find_optimal_market_conditions(
        self,
        tool_id: str,
        min_sample_size: int = 10,
        timeframe: Optional[str] = None,
        instrument: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Dict:
        """
        Find the optimal market conditions for a specific tool.

        Args:
            tool_id: Identifier for the trading tool
            min_sample_size: Minimum sample size for reliable analysis
            timeframe: Timeframe for analysis (e.g., '1h', '4h', 'D')
            instrument: Trading instrument (e.g., 'EUR_USD')
            from_date: Start date for analysis
            to_date: End date for analysis

        Returns:
            Optimal market conditions for the tool
        """
        params = {
            "tool_id": tool_id,
            "min_sample_size": min_sample_size,
            "timeframe": timeframe,
            "instrument": instrument
        }

        if from_date:
            params["from_date"] = from_date.isoformat()

        if to_date:
            params["to_date"] = to_date.isoformat()

        logger.info(f"Finding optimal market conditions for tool {tool_id}")
        return await self._make_request("GET", "/tools/optimal-conditions", params=params)

    async def analyze_tool_complementarity(
        self,
        tool_ids: List[str],
        timeframe: Optional[str] = None,
        instrument: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Dict:
        """
        Analyze how well different tools complement each other across market regimes.

        Args:
            tool_ids: List of tool identifiers to analyze
            timeframe: Timeframe for analysis (e.g., '1h', '4h', 'D')
            instrument: Trading instrument (e.g., 'EUR_USD')
            from_date: Start date for analysis
            to_date: End date for analysis

        Returns:
            Complementarity analysis for the tools
        """
        params = {
            "tool_ids": ",".join(tool_ids),
            "timeframe": timeframe,
            "instrument": instrument
        }

        if from_date:
            params["from_date"] = from_date.isoformat()

        if to_date:
            params["to_date"] = to_date.isoformat()

        logger.info(f"Analyzing tool complementarity for {len(tool_ids)} tools")
        return await self._make_request("GET", "/tools/complementarity", params=params)

    async def generate_performance_report(
        self,
        timeframe: Optional[str] = None,
        instrument: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Dict:
        """
        Generate a comprehensive performance report for all tools across market regimes.

        Args:
            timeframe: Timeframe for analysis (e.g., '1h', '4h', 'D')
            instrument: Trading instrument (e.g., 'EUR_USD')
            from_date: Start date for analysis
            to_date: End date for analysis

        Returns:
            Comprehensive performance report
        """
        params = {
            "timeframe": timeframe,
            "instrument": instrument
        }

        if from_date:
            params["from_date"] = from_date.isoformat()

        if to_date:
            params["to_date"] = to_date.isoformat()

        logger.info(f"Generating performance report for {instrument}/{timeframe}")
        return await self._make_request("GET", "/performance-report", params=params)

    async def recommend_tools_for_regime(
        self,
        current_regime: str,
        instrument: Optional[str] = None,
        timeframe: Optional[str] = None,
        min_sample_size: int = 10,
        min_win_rate: float = 50.0,
        top_n: int = 3
    ) -> Dict:
        """
        Recommend the best trading tools for the current market regime.

        Args:
            current_regime: Current market regime
            instrument: Trading instrument (e.g., 'EUR_USD')
            timeframe: Timeframe for analysis (e.g., '1h', '4h', 'D')
            min_sample_size: Minimum sample size for reliable analysis
            min_win_rate: Minimum win rate for recommended tools
            top_n: Number of top tools to recommend

        Returns:
            Recommended tools for the current regime
        """
        params = {
            "current_regime": current_regime,
            "instrument": instrument,
            "timeframe": timeframe,
            "min_sample_size": min_sample_size,
            "min_win_rate": min_win_rate,
            "top_n": top_n
        }

        logger.info(f"Recommending tools for {current_regime} regime")
        return await self._make_request("GET", "/tools/recommendations", params=params)

    async def analyze_effectiveness_trends(
        self,
        tool_id: str,
        timeframe: Optional[str] = None,
        instrument: Optional[str] = None,
        period_days: int = 30,
        look_back_periods: int = 6
    ) -> Dict:
        """
        Analyze how the effectiveness of a tool has changed over time across market regimes.

        Args:
            tool_id: Identifier for the trading tool
            timeframe: Timeframe for analysis (e.g., '1h', '4h', 'D')
            instrument: Trading instrument (e.g., 'EUR_USD')
            period_days: Number of days to analyze
            look_back_periods: Number of periods to look back

        Returns:
            Effectiveness trend analysis
        """
        params = {
            "tool_id": tool_id,
            "timeframe": timeframe,
            "instrument": instrument,
            "period_days": period_days,
            "look_back_periods": look_back_periods
        }

        logger.info(f"Analyzing effectiveness trends for tool {tool_id}")
        return await self._make_request("GET", "/tools/effectiveness-trends", params=params)

    async def get_underperforming_tools(
        self,
        win_rate_threshold: float = 50.0,
        min_sample_size: int = 20,
        timeframe: Optional[str] = None,
        instrument: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Dict:
        """
        Identify underperforming trading tools that may need optimization or retirement.

        Args:
            win_rate_threshold: Win rate threshold for underperforming tools
            min_sample_size: Minimum sample size for reliable analysis
            timeframe: Timeframe for analysis (e.g., '1h', '4h', 'D')
            instrument: Trading instrument (e.g., 'EUR_USD')
            from_date: Start date for analysis
            to_date: End date for analysis

        Returns:
            Underperforming tools analysis
        """
        params = {
            "win_rate_threshold": win_rate_threshold,
            "min_sample_size": min_sample_size,
            "timeframe": timeframe,
            "instrument": instrument
        }

        if from_date:
            params["from_date"] = from_date.isoformat()

        if to_date:
            params["to_date"] = to_date.isoformat()

        logger.info(f"Getting underperforming tools for {instrument}/{timeframe}")
        return await self._make_request("GET", "/tools/underperforming", params=params)
