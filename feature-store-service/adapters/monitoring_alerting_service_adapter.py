"""
Adapter implementation for monitoring-alerting-service service.
"""

from typing import Dict, List, Optional, Any
import requests
from common_lib.exceptions import ServiceException
from common_lib.resilience import circuit_breaker, retry_with_backoff
from ..interfaces.monitoring_alerting_service_interface import MonitoringAlertingServiceInterface


class MonitoringAlertingServiceAdapter(MonitoringAlertingServiceInterface):
    """
    Adapter for monitoring-alerting-service service.
    """

    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize the adapter.

        Args:
            base_url: Base URL of the service
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout

    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the service.

        Returns:
            Service status information
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/status",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ServiceException(f"Error getting service status: {str(e)}")

    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    def get_info(resource_id: str) -> Dict[str, Any]:
        """
        Get information from the service.

        Args:
            resource_id: Resource identifier
        Returns:
            Resource information
        """
        try:
            # Implement the method using requests to call the service API
            # This is a placeholder implementation
            response = requests.get(
                f"{self.base_url}/api/resources/" + str(resource_id) + "",
                params={},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ServiceException(f"Error calling get_info: {str(e)}")

    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    def list_resources(filter_params: Optional[Dict[str, Any]] = None, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """
        List available resources.

        Args:
            filter_params: Filter parameters
        Args:
            limit: Maximum number of results
        Args:
            offset: Result offset
        Returns:
            Dictionary with resources and pagination information
        """
        try:
            # Implement the method using requests to call the service API
            # This is a placeholder implementation
            response = requests.get(
                f"{self.base_url}/api/resources",
                params={'limit': limit, 'offset': offset, **(filter_params or {})},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ServiceException(f"Error calling list_resources: {str(e)}")

