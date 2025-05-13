"""
Adapter implementation for data-pipeline-service service.
"""

from typing import Dict, List, Optional, Any
import requests
from common_lib.exceptions import ServiceException
from common_lib.resilience import circuit_breaker, retry_with_backoff
from ..interfaces.data_pipeline_service_interface import DataPipelineServiceInterface


class DataPipelineServiceAdapter(DataPipelineServiceInterface):
    """
    Adapter for data-pipeline-service service.
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
    def get_data(dataset_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get data from the service.

        Args:
            dataset_id: Dataset identifier
        Args:
            start_date: Start date (ISO format)
        Args:
            end_date: End date (ISO format)
        Returns:
            List of data records
        """
        try:
            # Implement the method using requests to call the service API
            # This is a placeholder implementation
            response = requests.get(
                f"{self.base_url}/api/data",
                params={'dataset_id': dataset_id, 'start_date': start_date, 'end_date': end_date},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ServiceException(f"Error calling get_data: {str(e)}")

