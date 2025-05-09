"""
Standardized NLP Analysis Client

This module provides a client for interacting with the standardized NLP Analysis API.
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

class NLPAnalysisClient:
    """
    Client for interacting with the standardized NLP Analysis API.
    
    This client provides methods for analyzing news, economic reports,
    and generating combined insights for trading decisions.
    
    It includes resilience patterns like retry with backoff and circuit breaking.
    """
    
    def __init__(self, base_url: Optional[str] = None, timeout: int = 30):
        """
        Initialize the NLP Analysis client.
        
        Args:
            base_url: Base URL for the NLP Analysis API. If None, uses the URL from settings.
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.base_url = base_url or settings.analysis_engine_url
        self.timeout = timeout
        self.api_prefix = "/api/v1/analysis/nlp"
        
        # Configure circuit breaker
        self.circuit_breaker = circuit_breaker(
            failure_threshold=5,
            recovery_timeout=30,
            name="nlp_analysis_client"
        )
        
        logger.info(f"Initialized NLP Analysis client with base URL: {self.base_url}")
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Make a request to the NLP Analysis API with resilience patterns.
        
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
                            logger.error(f"Server error from NLP Analysis API: {error_text}")
                            raise ServiceUnavailableError(f"NLP Analysis API server error: {response.status}")
                        
                        if response.status >= 400:
                            error_text = await response.text()
                            logger.error(f"Client error from NLP Analysis API: {error_text}")
                            raise Exception(f"NLP Analysis API client error: {response.status} - {error_text}")
                        
                        return await response.json()
            except aiohttp.ClientError as e:
                logger.error(f"Connection error to NLP Analysis API: {str(e)}")
                raise ServiceUnavailableError(f"Failed to connect to NLP Analysis API: {str(e)}")
            except asyncio.TimeoutError:
                logger.error(f"Timeout connecting to NLP Analysis API")
                raise ServiceTimeoutError(f"Timeout connecting to NLP Analysis API")
        
        return await _request()
    
    async def analyze_news(
        self,
        news_items: List[Dict[str, Any]]
    ) -> Dict:
        """
        Analyze financial news content and assess potential market impact.
        
        Args:
            news_items: List of news items to analyze
            
        Returns:
            Analysis results
        """
        data = {
            "news_items": news_items
        }
        
        logger.info(f"Analyzing {len(news_items)} news items")
        return await self._make_request("POST", "/news/analyze", data=data)
    
    async def analyze_economic_report(
        self,
        report: Dict[str, Any]
    ) -> Dict:
        """
        Analyze economic report content and assess potential market impact.
        
        Args:
            report: Economic report to analyze
            
        Returns:
            Analysis results
        """
        data = {
            "report": report
        }
        
        logger.info(f"Analyzing economic report: {report.get('title', 'Unknown')}")
        return await self._make_request("POST", "/economic-reports/analyze", data=data)
    
    async def get_combined_insights(
        self,
        news_data: Optional[Dict[str, Any]] = None,
        economic_reports: Optional[List[Dict[str, Any]]] = None,
        currency_pairs: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate combined insights from news and economic reports.
        
        Args:
            news_data: News data to analyze
            economic_reports: Economic reports to analyze
            currency_pairs: Currency pairs to filter insights for
            
        Returns:
            Combined insights
        """
        data = {
            "news_data": news_data,
            "economic_reports": economic_reports,
            "currency_pairs": currency_pairs
        }
        
        logger.info(f"Getting combined insights for {len(currency_pairs) if currency_pairs else 'all'} currency pairs")
        return await self._make_request("POST", "/insights/combined", data=data)
    
    async def get_market_sentiment(
        self,
        currency_pair: Optional[str] = None
    ) -> Dict:
        """
        Get current market sentiment based on recent news and economic reports.
        
        Args:
            currency_pair: Specific currency pair to analyze
            
        Returns:
            Market sentiment analysis
        """
        params = {}
        if currency_pair:
            params["currency_pair"] = currency_pair
        
        logger.info(f"Getting market sentiment for {currency_pair if currency_pair else 'all currency pairs'}")
        return await self._make_request("GET", "/market-sentiment", params=params)
