"""
Interface definition for portfolio-management-service service.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class PortfolioManagementServiceInterface(ABC):
    """
    Interface for portfolio-management-service service.
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
    def get_info(resource_id: str) -> Dict[str, Any]:
        """
        Get information from the service.

        Args:
            resource_id: Resource identifier
        Returns:
            Resource information
        """
        pass

    @abstractmethod
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
        pass

