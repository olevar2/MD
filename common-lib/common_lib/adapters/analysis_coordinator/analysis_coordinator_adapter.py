"""
Adapter for the Analysis Coordinator Service.

This module provides an adapter for the Analysis Coordinator Service, implementing
the IAnalysisCoordinatorService interface.
"""
from typing import Dict, Any, List, Optional
import logging
import httpx
from datetime import datetime
from common_lib.interfaces.analysis_coordinator_service_interface import IAnalysisCoordinatorService
from common_lib.resilience.decorators import with_circuit_breaker, with_retry, with_timeout
from common_lib.resilience.factory import create_standard_resilience_config

logger = logging.getLogger(__name__)


class AnalysisCoordinatorAdapter(IAnalysisCoordinatorService):
    """Adapter for the Analysis Coordinator Service."""

    def __init__(self, base_url: str, timeout: float = 60.0):
        """
        Initialize the AnalysisCoordinatorAdapter.

        Args:
            base_url: The base URL of the Analysis Coordinator Service
            timeout: The timeout for HTTP requests in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.resilience_config = create_standard_resilience_config(
            service_name="analysis-coordinator-service",
            timeout_seconds=timeout
        )
        logger.info(f"Initialized AnalysisCoordinatorAdapter with base URL: {base_url}")

    @with_circuit_breaker("analysis-coordinator-service")
    @with_retry("analysis-coordinator-service")
    @with_timeout("analysis-coordinator-service")
    async def run_integrated_analysis(self, 
                                     symbol: str,
                                     timeframe: str,
                                     data: Dict[str, Any],
                                     analysis_types: List[str],
                                     config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run integrated analysis across multiple analysis services.

        Args:
            symbol: The symbol to analyze
            timeframe: The timeframe to analyze
            data: The market data to analyze
            analysis_types: List of analysis types to perform
            config: Optional configuration parameters

        Returns:
            A dictionary containing the integrated analysis results
        """
        url = f"{self.base_url}/api/v1/integrated-analysis"
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": data,
            "analysis_types": analysis_types,
            "config": config or {}
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("analysis-coordinator-service")
    @with_retry("analysis-coordinator-service")
    @with_timeout("analysis-coordinator-service")
    async def create_analysis_task(self, 
                                  task_type: str,
                                  parameters: Dict[str, Any],
                                  priority: Optional[int] = 1,
                                  callback_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new analysis task to be executed asynchronously.

        Args:
            task_type: The type of analysis task to create
            parameters: The parameters for the analysis task
            priority: Optional priority for the task (higher number = higher priority)
            callback_url: Optional URL to call when the task is complete

        Returns:
            A dictionary containing the created task information
        """
        url = f"{self.base_url}/api/v1/tasks"
        payload = {
            "task_type": task_type,
            "parameters": parameters,
            "priority": priority
        }
        
        if callback_url:
            payload["callback_url"] = callback_url
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("analysis-coordinator-service")
    @with_retry("analysis-coordinator-service")
    @with_timeout("analysis-coordinator-service")
    async def get_task_status(self, 
                             task_id: str) -> Dict[str, Any]:
        """
        Get the status of an analysis task.

        Args:
            task_id: The ID of the task to check

        Returns:
            A dictionary containing the task status information
        """
        url = f"{self.base_url}/api/v1/tasks/{task_id}/status"
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("analysis-coordinator-service")
    @with_retry("analysis-coordinator-service")
    @with_timeout("analysis-coordinator-service")
    async def get_task_result(self, 
                             task_id: str) -> Dict[str, Any]:
        """
        Get the result of a completed analysis task.

        Args:
            task_id: The ID of the task to get the result for

        Returns:
            A dictionary containing the task result
        """
        url = f"{self.base_url}/api/v1/tasks/{task_id}/result"
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("analysis-coordinator-service")
    @with_retry("analysis-coordinator-service")
    @with_timeout("analysis-coordinator-service")
    async def cancel_task(self, 
                         task_id: str) -> Dict[str, Any]:
        """
        Cancel a running analysis task.

        Args:
            task_id: The ID of the task to cancel

        Returns:
            A dictionary containing the cancellation result
        """
        url = f"{self.base_url}/api/v1/tasks/{task_id}/cancel"
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("analysis-coordinator-service")
    @with_retry("analysis-coordinator-service")
    @with_timeout("analysis-coordinator-service")
    async def list_tasks(self,
                        status: Optional[str] = None,
                        task_type: Optional[str] = None,
                        created_after: Optional[datetime] = None,
                        created_before: Optional[datetime] = None,
                        limit: int = 100,
                        offset: int = 0) -> Dict[str, Any]:
        """
        List analysis tasks with optional filtering.

        Args:
            status: Optional status filter
            task_type: Optional task type filter
            created_after: Optional creation date filter (after)
            created_before: Optional creation date filter (before)
            limit: Maximum number of results to return
            offset: Offset for pagination

        Returns:
            A dictionary containing the list of tasks and pagination information
        """
        url = f"{self.base_url}/api/v1/tasks"
        params = {
            "limit": limit,
            "offset": offset
        }
        
        if status:
            params["status"] = status
        if task_type:
            params["task_type"] = task_type
        if created_after:
            params["created_after"] = created_after.isoformat()
        if created_before:
            params["created_before"] = created_before.isoformat()
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("analysis-coordinator-service")
    @with_retry("analysis-coordinator-service")
    @with_timeout("analysis-coordinator-service")
    async def run_multi_timeframe_analysis(self,
                                          symbol: str,
                                          timeframes: List[str],
                                          data: Dict[str, Dict[str, Any]],
                                          analysis_types: List[str],
                                          config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run analysis across multiple timeframes and consolidate the results.

        Args:
            symbol: The symbol to analyze
            timeframes: List of timeframes to analyze
            data: Dictionary of market data for each timeframe
            analysis_types: List of analysis types to perform
            config: Optional configuration parameters

        Returns:
            A dictionary containing the multi-timeframe analysis results
        """
        url = f"{self.base_url}/api/v1/multi-timeframe-analysis"
        payload = {
            "symbol": symbol,
            "timeframes": timeframes,
            "data": data,
            "analysis_types": analysis_types,
            "config": config or {}
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("analysis-coordinator-service")
    @with_retry("analysis-coordinator-service")
    @with_timeout("analysis-coordinator-service")
    async def run_multi_symbol_analysis(self,
                                       symbols: List[str],
                                       timeframe: str,
                                       data: Dict[str, Dict[str, Any]],
                                       analysis_types: List[str],
                                       config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run analysis across multiple symbols and consolidate the results.

        Args:
            symbols: List of symbols to analyze
            timeframe: The timeframe to analyze
            data: Dictionary of market data for each symbol
            analysis_types: List of analysis types to perform
            config: Optional configuration parameters

        Returns:
            A dictionary containing the multi-symbol analysis results
        """
        url = f"{self.base_url}/api/v1/multi-symbol-analysis"
        payload = {
            "symbols": symbols,
            "timeframe": timeframe,
            "data": data,
            "analysis_types": analysis_types,
            "config": config or {}
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()