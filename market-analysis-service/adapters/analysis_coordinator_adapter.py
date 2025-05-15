"""
Analysis Coordinator Adapter for Market Analysis Service.

This module provides an adapter for communicating with the Analysis Coordinator Service
to coordinate analysis tasks.
"""
import logging
import uuid
from typing import Dict, List, Any, Optional
import httpx
from common_lib.resilience.decorators import (
    retry_with_backoff,
    circuit_breaker,
    timeout
)

logger = logging.getLogger(__name__)

class AnalysisCoordinatorAdapter:
    """
    Adapter for communicating with the Analysis Coordinator Service.
    """
    
    def __init__(self, base_url: str = "http://analysis-coordinator-service:8000"):
        """
        Initialize the Analysis Coordinator Adapter.
        
        Args:
            base_url: Base URL of the Analysis Coordinator Service
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        
    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @timeout(seconds=10)
    async def register_analysis_task(
        self,
        task_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Register an analysis task with the Analysis Coordinator Service.
        
        Args:
            task_type: Type of analysis task
            parameters: Task parameters
            
        Returns:
            Task registration response
        """
        try:
            request_id = str(uuid.uuid4())
            headers = {"X-Request-ID": request_id}
            
            payload = {
                "task_type": task_type,
                "parameters": parameters,
                "source_service": "market-analysis-service"
            }
            
            url = f"{self.base_url}/api/v1/coordinator/tasks"
            
            logger.info(f"Registering {task_type} analysis task")
            response = await self.client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error when registering analysis task: {e}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error when registering analysis task: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error when registering analysis task: {e}")
            raise
            
    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @timeout(seconds=10)
    async def get_task_status(
        self,
        task_id: str
    ) -> Dict[str, Any]:
        """
        Get the status of an analysis task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task status
        """
        try:
            request_id = str(uuid.uuid4())
            headers = {"X-Request-ID": request_id}
            
            url = f"{self.base_url}/api/v1/coordinator/tasks/{task_id}"
            
            logger.info(f"Getting status for task {task_id}")
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error when getting task status: {e}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error when getting task status: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error when getting task status: {e}")
            raise
            
    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @timeout(seconds=10)
    async def get_task_result(
        self,
        task_id: str
    ) -> Dict[str, Any]:
        """
        Get the result of an analysis task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task result
        """
        try:
            request_id = str(uuid.uuid4())
            headers = {"X-Request-ID": request_id}
            
            url = f"{self.base_url}/api/v1/coordinator/tasks/{task_id}/result"
            
            logger.info(f"Getting result for task {task_id}")
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error when getting task result: {e}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error when getting task result: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error when getting task result: {e}")
            raise