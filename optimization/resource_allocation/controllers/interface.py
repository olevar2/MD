"""
Resource Controller Interface

This module defines the interface for resource controllers that apply resource
allocation decisions to the underlying infrastructure.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

from ..models import ResourceType


class ResourceControllerInterface(ABC):
    """Interface for resource controllers."""

    @abstractmethod
    def update_resources(self, service_name: str, resources: Dict[ResourceType, float]) -> bool:
        """
        Update resources for a service.

        Args:
            service_name: Name of the service
            resources: Resource allocations

        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_current_resources(self, service_name: str) -> Optional[Dict[ResourceType, float]]:
        """
        Get current resource allocations for a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Dictionary of current resource allocations or None if not available
        """
        pass
    
    @abstractmethod
    def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """
        Get status information for a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Dictionary with service status information
        """
        pass