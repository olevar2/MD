"""
Interface definition for data-management-service service.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class DataManagementServiceInterface(ABC):
    """
    Interface for data-management-service service.
    """

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the service.

        Returns:
            Service status information
        """
        pass

    @abstractmethod
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
        pass

