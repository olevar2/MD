"""
Standardized Feedback Client

This module provides a client for interacting with the standardized Feedback API.
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

class FeedbackClient:
    """
    Client for interacting with the standardized Feedback API.
    
    This client provides methods for accessing feedback-related functionality,
    including retrieving feedback statistics, triggering model retraining,
    and managing feedback rules.
    
    It includes resilience patterns like retry with backoff and circuit breaking.
    """
    
    def __init__(self, base_url: Optional[str] = None, timeout: int = 30):
        """
        Initialize the Feedback client.
        
        Args:
            base_url: Base URL for the Feedback API. If None, uses the URL from settings.
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.base_url = base_url or settings.analysis_engine_url
        self.timeout = timeout
        self.api_prefix = "/api/v1/analysis/feedback"
        
        # Configure circuit breaker
        self.circuit_breaker = circuit_breaker(
            failure_threshold=5,
            recovery_timeout=30,
            name="feedback_client"
        )
        
        logger.info(f"Initialized Feedback client with base URL: {self.base_url}")
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Make a request to the Feedback API with resilience patterns.
        
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
                            logger.error(f"Server error from Feedback API: {error_text}")
                            raise ServiceUnavailableError(f"Feedback API server error: {response.status}")
                        
                        if response.status >= 400:
                            error_text = await response.text()
                            logger.error(f"Client error from Feedback API: {error_text}")
                            raise Exception(f"Feedback API client error: {response.status} - {error_text}")
                        
                        return await response.json()
            except aiohttp.ClientError as e:
                logger.error(f"Connection error to Feedback API: {str(e)}")
                raise ServiceUnavailableError(f"Failed to connect to Feedback API: {str(e)}")
            except asyncio.TimeoutError:
                logger.error(f"Timeout connecting to Feedback API")
                raise ServiceTimeoutError(f"Timeout connecting to Feedback API")
        
        return await _request()
    
    async def get_feedback_statistics(
        self,
        strategy_id: Optional[str] = None,
        model_id: Optional[str] = None,
        instrument: Optional[str] = None,
        start_time: Optional[Union[datetime, str]] = None,
        end_time: Optional[Union[datetime, str]] = None
    ) -> Dict:
        """
        Get feedback statistics with optional filtering.
        
        Args:
            strategy_id: Filter by strategy ID
            model_id: Filter by model ID
            instrument: Filter by instrument
            start_time: Start time for filtering (ISO format)
            end_time: End time for filtering (ISO format)
            
        Returns:
            Feedback statistics
        """
        params = {}
        if strategy_id:
            params["strategy_id"] = strategy_id
        if model_id:
            params["model_id"] = model_id
        if instrument:
            params["instrument"] = instrument
        if start_time:
            params["start_time"] = start_time.isoformat() if isinstance(start_time, datetime) else start_time
        if end_time:
            params["end_time"] = end_time.isoformat() if isinstance(end_time, datetime) else end_time
        
        logger.info(f"Getting feedback statistics for strategy {strategy_id if strategy_id else 'all strategies'}")
        return await self._make_request("GET", "/statistics", params=params)
    
    async def trigger_model_retraining(
        self,
        model_id: str
    ) -> Dict:
        """
        Trigger retraining of a specific model based on collected feedback.
        
        Args:
            model_id: ID of the model to retrain
            
        Returns:
            Retraining status
        """
        logger.info(f"Triggering retraining for model {model_id}")
        return await self._make_request("POST", f"/models/{model_id}/retrain")
    
    async def update_feedback_rules(
        self,
        rule_updates: List[Dict[str, Any]]
    ) -> Dict:
        """
        Update feedback categorization rules.
        
        Args:
            rule_updates: List of rule updates to apply
            
        Returns:
            Rule update status
        """
        data = {
            "updates": rule_updates
        }
        
        logger.info(f"Updating {len(rule_updates)} feedback rules")
        return await self._make_request("PUT", "/rules", data=data)
    
    async def get_parameter_performance(
        self,
        strategy_id: str,
        min_samples: int = 10
    ) -> Dict:
        """
        Get performance statistics for strategy parameters.
        
        Args:
            strategy_id: ID of the strategy to analyze
            min_samples: Minimum number of samples required for parameter statistics
            
        Returns:
            Parameter performance statistics
        """
        params = {
            "min_samples": min_samples
        }
        
        logger.info(f"Getting parameter performance for strategy {strategy_id}")
        return await self._make_request("GET", f"/strategies/{strategy_id}/parameters", params=params)
    
    async def submit_feedback(
        self,
        source: str,
        target_id: str,
        feedback_type: str,
        content: Optional[Dict[str, Any]] = None,
        timestamp: Optional[Union[datetime, str]] = None
    ) -> Dict:
        """
        Submit feedback for a signal, model, or strategy.
        
        Args:
            source: Source of the feedback (e.g., 'user', 'system')
            target_id: ID of the target (signal, model, strategy)
            feedback_type: Type of feedback (e.g., 'accuracy', 'usefulness')
            content: Additional feedback content
            timestamp: When the feedback was generated
            
        Returns:
            Feedback submission status
        """
        data = {
            "source": source,
            "target_id": target_id,
            "feedback_type": feedback_type
        }
        
        if content:
            data["content"] = content
        
        if timestamp:
            data["timestamp"] = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
        
        logger.info(f"Submitting feedback for {target_id}")
        return await self._make_request("POST", "/submit", data=data)
