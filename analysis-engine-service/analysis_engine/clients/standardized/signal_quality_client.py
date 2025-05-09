"""
Standardized Signal Quality Client

This module provides a client for interacting with the standardized Signal Quality API.
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

class SignalQualityClient:
    """
    Client for interacting with the standardized Signal Quality API.
    
    This client provides methods for evaluating signal quality,
    analyzing the relationship between signal quality and outcomes,
    and tracking quality trends over time.
    
    It includes resilience patterns like retry with backoff and circuit breaking.
    """
    
    def __init__(self, base_url: Optional[str] = None, timeout: int = 30):
        """
        Initialize the Signal Quality client.
        
        Args:
            base_url: Base URL for the Signal Quality API. If None, uses the URL from settings.
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.base_url = base_url or settings.analysis_engine_url
        self.timeout = timeout
        self.api_prefix = "/api/v1/analysis/signal-quality"
        
        # Configure circuit breaker
        self.circuit_breaker = circuit_breaker(
            failure_threshold=5,
            recovery_timeout=30,
            name="signal_quality_client"
        )
        
        logger.info(f"Initialized Signal Quality client with base URL: {self.base_url}")
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Make a request to the Signal Quality API with resilience patterns.
        
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
                            logger.error(f"Server error from Signal Quality API: {error_text}")
                            raise ServiceUnavailableError(f"Signal Quality API server error: {response.status}")
                        
                        if response.status >= 400:
                            error_text = await response.text()
                            logger.error(f"Client error from Signal Quality API: {error_text}")
                            raise Exception(f"Signal Quality API client error: {response.status} - {error_text}")
                        
                        return await response.json()
            except aiohttp.ClientError as e:
                logger.error(f"Connection error to Signal Quality API: {str(e)}")
                raise ServiceUnavailableError(f"Failed to connect to Signal Quality API: {str(e)}")
            except asyncio.TimeoutError:
                logger.error(f"Timeout connecting to Signal Quality API")
                raise ServiceTimeoutError(f"Timeout connecting to Signal Quality API")
        
        return await _request()
    
    async def evaluate_signal_quality(
        self,
        signal_id: str,
        market_context: Optional[Dict[str, Any]] = None,
        historical_data: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Evaluate the quality of a specific trading signal.
        
        Args:
            signal_id: ID of the signal to evaluate
            market_context: Additional market context
            historical_data: Historical performance data
            
        Returns:
            Signal quality evaluation
        """
        data = {
            "signal_id": signal_id,
            "market_context": market_context or {},
            "historical_data": historical_data or {}
        }
        
        logger.info(f"Evaluating quality for signal {signal_id}")
        return await self._make_request("POST", f"/signals/{signal_id}/evaluate", data=data)
    
    async def analyze_signal_quality(
        self,
        tool_id: Optional[str] = None,
        timeframe: Optional[str] = None,
        market_regime: Optional[str] = None,
        days: Optional[int] = 30
    ) -> Dict:
        """
        Analyze the relationship between signal quality and outcomes.
        
        Args:
            tool_id: Filter by specific tool ID
            timeframe: Filter by specific timeframe
            market_regime: Filter by market regime
            days: Number of days to analyze
            
        Returns:
            Signal quality analysis
        """
        data = {
            "tool_id": tool_id,
            "timeframe": timeframe,
            "market_regime": market_regime,
            "days": days
        }
        
        logger.info(f"Analyzing signal quality for tool {tool_id}, timeframe {timeframe}, market regime {market_regime}")
        return await self._make_request("POST", "/analyze", data=data)
    
    async def analyze_quality_trends(
        self,
        tool_id: str,
        window_size: int = 20,
        days: int = 90
    ) -> Dict:
        """
        Analyze trends in signal quality over time for a specific tool.
        
        Args:
            tool_id: Tool ID to analyze
            window_size: Size of moving window for trend analysis
            days: Number of days to analyze
            
        Returns:
            Quality trend analysis
        """
        data = {
            "tool_id": tool_id,
            "window_size": window_size,
            "days": days
        }
        
        logger.info(f"Analyzing quality trends for tool {tool_id}")
        return await self._make_request("POST", "/trends", data=data)
