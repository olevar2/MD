"""
Standardized Adaptive Layer Client

This module provides a client for interacting with the standardized Adaptive Layer API.
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

class AdaptiveLayerClient:
    """
    Client for interacting with the standardized Adaptive Layer API.

    This client provides methods for generating adaptive parameters,
    adjusting parameters, updating strategy parameters, and more.

    It includes resilience patterns like retry with backoff and circuit breaking.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: int = 30):
        """
        Initialize the Adaptive Layer client.

        Args:
            base_url: Base URL for the Adaptive Layer API. If None, uses the URL from settings.
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.base_url = base_url or settings.analysis_engine_url
        self.timeout = timeout
        self.api_prefix = "/api/v1/analysis/adaptations"

        # Configure circuit breaker
        self.circuit_breaker = circuit_breaker(
            failure_threshold=5,
            recovery_timeout=30,
            name="adaptive_layer_client"
        )

        logger.info(f"Initialized Adaptive Layer client with base URL: {self.base_url}")

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Make a request to the Adaptive Layer API with resilience patterns.

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
                            logger.error(f"Server error from Adaptive Layer API: {error_text}")
                            raise ServiceUnavailableError(f"Adaptive Layer API server error: {response.status}")

                        if response.status >= 400:
                            error_text = await response.text()
                            logger.error(f"Client error from Adaptive Layer API: {error_text}")
                            raise Exception(f"Adaptive Layer API client error: {response.status} - {error_text}")

                        return await response.json()
            except aiohttp.ClientError as e:
                logger.error(f"Connection error to Adaptive Layer API: {str(e)}")
                raise ServiceUnavailableError(f"Failed to connect to Adaptive Layer API: {str(e)}")
            except asyncio.TimeoutError:
                logger.error(f"Timeout connecting to Adaptive Layer API")
                raise ServiceTimeoutError(f"Timeout connecting to Adaptive Layer API")

        return await _request()

    async def generate_adaptive_parameters(
        self,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        ohlc_data: Union[List[Dict], pd.DataFrame],
        available_tools: List[str],
        adaptation_strategy: str = "moderate"
    ) -> Dict:
        """
        Generate adaptive parameters based on current market conditions and tool effectiveness.

        Args:
            strategy_id: Identifier for the strategy
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe for analysis (e.g., '1h', '4h', 'D')
            ohlc_data: OHLC price data as list of dicts or DataFrame
            available_tools: List of available tools
            adaptation_strategy: Adaptation strategy ('conservative', 'moderate', 'aggressive')

        Returns:
            Generated parameters
        """
        # Convert DataFrame to list of dicts if needed
        if isinstance(ohlc_data, pd.DataFrame):
            if 'timestamp' not in ohlc_data.columns and ohlc_data.index.name != 'timestamp':
                ohlc_data = ohlc_data.reset_index()

            ohlc_data = ohlc_data.to_dict('records')

        data = {
            "strategy_id": strategy_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "ohlc_data": ohlc_data,
            "available_tools": available_tools,
            "adaptation_strategy": adaptation_strategy
        }

        logger.info(f"Generating adaptive parameters for strategy {strategy_id}, symbol {symbol}, timeframe {timeframe}")
        return await self._make_request("POST", "/parameters/generate", data=data)

    async def adjust_parameters(
        self,
        strategy_id: str,
        instrument: str,
        timeframe: str,
        current_parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Adjust strategy parameters based on market conditions and tool effectiveness.

        Args:
            strategy_id: Identifier for the strategy
            instrument: Trading instrument (e.g., 'EUR_USD')
            timeframe: Timeframe for analysis (e.g., '1H', '4H', 'D')
            current_parameters: Current strategy parameters
            context: Additional context information (e.g., market data, regime)

        Returns:
            Adjusted parameters response
        """
        data = {
            "strategy_id": strategy_id,
            "instrument": instrument,
            "timeframe": timeframe,
            "current_parameters": current_parameters,
            "context": context or {}
        }

        logger.info(f"Adjusting parameters for strategy {strategy_id}, instrument {instrument}, timeframe {timeframe}")
        return await self._make_request("POST", "/parameters/adjust", data=data)

    async def update_strategy_parameters(
        self,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        ohlc_data: Union[List[Dict], pd.DataFrame],
        available_tools: List[str],
        adaptation_strategy: str = "moderate",
        strategy_execution_api_url: Optional[str] = None
    ) -> Dict:
        """
        Generate adaptive parameters and apply them to the strategy execution engine.

        Args:
            strategy_id: Identifier for the strategy
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe for analysis (e.g., '1h', '4h', 'D')
            ohlc_data: OHLC price data as list of dicts or DataFrame
            available_tools: List of available tools
            adaptation_strategy: Adaptation strategy ('conservative', 'moderate', 'aggressive')
            strategy_execution_api_url: Optional URL for the strategy execution API

        Returns:
            Updated strategy parameters
        """
        # Convert DataFrame to list of dicts if needed
        if isinstance(ohlc_data, pd.DataFrame):
            if 'timestamp' not in ohlc_data.columns and ohlc_data.index.name != 'timestamp':
                ohlc_data = ohlc_data.reset_index()

            ohlc_data = ohlc_data.to_dict('records')

        data = {
            "strategy_id": strategy_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "ohlc_data": ohlc_data,
            "available_tools": available_tools,
            "adaptation_strategy": adaptation_strategy,
            "strategy_execution_api_url": strategy_execution_api_url
        }

        logger.info(f"Updating strategy parameters for strategy {strategy_id}, symbol {symbol}, timeframe {timeframe}")
        return await self._make_request("POST", "/strategy/update", data=data)

    async def generate_strategy_recommendations(
        self,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        ohlc_data: Union[List[Dict], pd.DataFrame],
        current_tools: List[str],
        all_available_tools: List[str]
    ) -> Dict:
        """
        Generate recommendations for optimizing a strategy based on effectiveness data.

        Args:
            strategy_id: Identifier for the strategy
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe for analysis (e.g., '1h', '4h', 'D')
            ohlc_data: OHLC price data as list of dicts or DataFrame
            current_tools: List of currently used tools
            all_available_tools: List of all available tools

        Returns:
            Strategy recommendations
        """
        # Convert DataFrame to list of dicts if needed
        if isinstance(ohlc_data, pd.DataFrame):
            if 'timestamp' not in ohlc_data.columns and ohlc_data.index.name != 'timestamp':
                ohlc_data = ohlc_data.reset_index()

            ohlc_data = ohlc_data.to_dict('records')

        data = {
            "strategy_id": strategy_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "ohlc_data": ohlc_data,
            "current_tools": current_tools,
            "all_available_tools": all_available_tools
        }

        logger.info(f"Generating strategy recommendations for strategy {strategy_id}, symbol {symbol}, timeframe {timeframe}")
        return await self._make_request("POST", "/strategy/recommendations", data=data)

    async def analyze_strategy_effectiveness_trend(
        self,
        strategy_id: str,
        symbol: str,
        timeframe: str,
        period_days: int = 30,
        look_back_periods: int = 6
    ) -> Dict:
        """
        Analyze how a strategy's effectiveness has changed over time.

        Args:
            strategy_id: Identifier for the strategy
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe for analysis (e.g., '1h', '4h', 'D')
            period_days: Number of days to analyze
            look_back_periods: Number of periods to look back

        Returns:
            Strategy effectiveness trend analysis
        """
        data = {
            "strategy_id": strategy_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "period_days": period_days,
            "look_back_periods": look_back_periods
        }

        logger.info(f"Analyzing strategy effectiveness trend for strategy {strategy_id}, symbol {symbol}, timeframe {timeframe}")
        return await self._make_request("POST", "/strategy/effectiveness-trend", data=data)

    async def record_strategy_outcome(
        self,
        strategy_id: str,
        instrument: str,
        timeframe: str,
        adaptation_id: str,
        outcome_metrics: Dict[str, Any],
        market_regime: Optional[str] = None,
        feedback_content: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Record the outcome of a strategy execution with adapted parameters.

        Args:
            strategy_id: Identifier for the strategy
            instrument: Trading instrument (e.g., 'EUR_USD')
            timeframe: Timeframe for analysis (e.g., '1H', '4H', 'D')
            adaptation_id: Identifier for the specific adaptation being evaluated
            outcome_metrics: Performance metrics from strategy execution (e.g., pnl, win_rate)
            market_regime: Market regime during execution (if known)
            feedback_content: Additional qualitative or quantitative feedback

        Returns:
            Status response
        """
        data = {
            "strategy_id": strategy_id,
            "instrument": instrument,
            "timeframe": timeframe,
            "adaptation_id": adaptation_id,
            "outcome_metrics": outcome_metrics,
            "market_regime": market_regime,
            "feedback_content": feedback_content
        }

        logger.info(f"Recording strategy outcome for strategy {strategy_id}, instrument {instrument}, timeframe {timeframe}")
        return await self._make_request("POST", "/feedback/outcomes", data=data)

    async def get_adaptation_history(self) -> Dict:
        """
        Get the history of adaptation decisions from the adaptation engine.

        Returns:
            Adaptation history
        """
        logger.info("Getting adaptation history")
        return await self._make_request("GET", "/adaptations/history")

    async def get_parameter_history(
        self,
        strategy_id: str,
        instrument: str,
        timeframe: str,
        limit: int = 10
    ) -> Dict:
        """
        Get parameter adjustment history for a specific strategy, instrument, and timeframe.

        Args:
            strategy_id: Identifier for the strategy
            instrument: Trading instrument
            timeframe: Timeframe for analysis
            limit: Maximum number of history entries to return

        Returns:
            Parameter history
        """
        params = {"limit": limit}

        # Convert parameters to kebab-case
        strategy_id_kebab = strategy_id.replace("_", "-")
        instrument_kebab = instrument.replace("_", "-")
        timeframe_kebab = timeframe.replace("_", "-")

        endpoint = f"/parameters/history/{strategy_id_kebab}/{instrument_kebab}/{timeframe_kebab}"

        logger.info(f"Getting parameter history for strategy {strategy_id}, instrument {instrument}, timeframe {timeframe}")
        return await self._make_request("GET", endpoint, params=params)

    async def get_adaptation_insights(self, strategy_id: str) -> Dict:
        """
        Generate insights from feedback data for a specific strategy.

        Args:
            strategy_id: Identifier for the strategy

        Returns:
            Adaptation insights
        """
        # Convert strategy_id to kebab-case
        strategy_id_kebab = strategy_id.replace("_", "-")
        endpoint = f"/feedback/insights/{strategy_id_kebab}"

        logger.info(f"Getting adaptation insights for strategy {strategy_id}")
        return await self._make_request("GET", endpoint)

    async def get_performance_by_regime(
        self,
        strategy_id: str,
        market_regime: Optional[str] = None
    ) -> Dict:
        """
        Get aggregated performance metrics by market regime for a specific strategy.

        Args:
            strategy_id: Identifier for the strategy
            market_regime: Optional filter for a specific market regime

        Returns:
            Performance metrics by regime
        """
        params = {}
        if market_regime:
            params["market_regime"] = market_regime

        # Convert strategy_id to kebab-case
        strategy_id_kebab = strategy_id.replace("_", "-")
        endpoint = f"/feedback/performance/{strategy_id_kebab}"

        logger.info(f"Getting performance by regime for strategy {strategy_id}")
        return await self._make_request("GET", endpoint, params=params)
