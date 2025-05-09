"""
Metrics Provider Interface

This module defines the interface for metrics providers that collect resource
utilization data from services.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List

from ..models import ResourceUtilization


class MetricsProviderInterface(ABC):
    """Interface for metrics providers."""

    @abstractmethod
    def get_metrics(self, service_name: str) -> Optional[Dict[str, float]]:
        """
        Get metrics for a service.

        Args:
            service_name: Name of the service

        Returns:
            Dictionary with metrics or None if unavailable
        """
        pass
    
    @abstractmethod
    def get_resource_utilization(self, service_name: str) -> Optional[ResourceUtilization]:
        """
        Get resource utilization for a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            ResourceUtilization object or None if unavailable
        """
        pass
    
    @abstractmethod
    def get_historical_metrics(
        self, 
        service_name: str, 
        metric_name: str, 
        start_time: str, 
        end_time: str, 
        step: str = "1m"
    ) -> List[Dict[str, Any]]:
        """
        Get historical metrics for a service.
        
        Args:
            service_name: Name of the service
            metric_name: Name of the metric to retrieve
            start_time: Start time in ISO format
            end_time: End time in ISO format
            step: Time step for data points
            
        Returns:
            List of metric data points
        """
        pass