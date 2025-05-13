"""
Service-Specific Performance Monitoring

This module provides service-specific performance monitoring for the service.
It extends the common-lib performance monitoring with service-specific metrics and operations.

To use this template:
1. Copy this file to your service's monitoring directory
2. Replace SERVICE_NAME with your service name
3. Add service-specific operations and metrics
"""

import os
import time
import logging
import functools
from typing import Dict, List, Any, Optional, Callable, Union, TypeVar, cast
from contextlib import contextmanager

from common_lib.monitoring.performance_monitoring import (
    PerformanceMonitor,
    track_operation as base_track_operation,
    track_performance as base_track_performance,
    get_performance_monitor
)

from common_lib.errors.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    BaseError,
    ServiceError
)

# Type variable for function
F = TypeVar('F', bound=Callable[..., Any])

# Create logger
logger = logging.getLogger(__name__)

# Service name
SERVICE_NAME = "portfolio-management"  # Replace with your service name

# Service-specific component names
COMPONENT_NAMES = {
    "portfolio": "Portfolio Management",
    "position": "Position Management",
    "risk": "Risk Management",
    "allocation": "Asset Allocation",
    "rebalancing": "Portfolio Rebalancing",
    "reporting": "Performance Reporting",
    "api": "API Endpoints",
    "service": "Service Layer",
    "repository": "Data Access",
    "client": "External Clients",
    "processor": "Data Processing",
    "validation": "Data Validation"
}


class ServicePerformanceMonitor:
    """
    Service-specific performance monitoring.
    
    This class extends the common-lib performance monitoring with service-specific
    metrics and operations.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'ServicePerformanceMonitor':
        """
        Get the singleton instance of the service performance monitor.
        
        Returns:
            ServicePerformanceMonitor instance
        """
        if cls._instance is None:
            cls._instance = ServicePerformanceMonitor()
        return cls._instance
    
    def __init__(self):
        """Initialize the service performance monitor."""
        # Get the base performance monitor
        self.monitor = get_performance_monitor()
        
        # Set service name
        if self.monitor.service_name == "unknown":
            self.monitor.service_name = SERVICE_NAME
        
        # Initialize service-specific metrics
        self._init_service_metrics()
    
    def _init_service_metrics(self):
        """Initialize service-specific metrics."""
        # Add service-specific metrics here
        pass
    
    @contextmanager
    def track_api_operation(self, operation: str):
        """
        Track an API operation.
        
        Args:
            operation: Operation name
            
        Yields:
            None
        """
        with self.monitor.track_operation("api", operation):
            yield
    
    @contextmanager
    def track_service_operation(self, operation: str):
        """
        Track a service operation.
        
        Args:
            operation: Operation name
            
        Yields:
            None
        """
        with self.monitor.track_operation("service", operation):
            yield
    
    @contextmanager
    def track_repository_operation(self, operation: str):
        """
        Track a repository operation.
        
        Args:
            operation: Operation name
            
        Yields:
            None
        """
        with self.monitor.track_operation("repository", operation):
            yield
    
    @contextmanager
    def track_client_operation(self, operation: str):
        """
        Track a client operation.
        
        Args:
            operation: Operation name
            
        Yields:
            None
        """
        with self.monitor.track_operation("client", operation):
            yield
    
    @contextmanager
    def track_processor_operation(self, operation: str):
        """
        Track a processor operation.
        
        Args:
            operation: Operation name
            
        Yields:
            None
        """
        with self.monitor.track_operation("processor", operation):
            yield
    
    @contextmanager
    def track_validation_operation(self, operation: str):
        """
        Track a validation operation.
        
        Args:
            operation: Operation name
            
        Yields:
            None
        """
        with self.monitor.track_operation("validation", operation):
            yield
    
    def track_api_function(self, operation: Optional[str] = None) -> Callable[[F], F]:
        """
        Decorator for tracking an API function.
        
        Args:
            operation: Operation name (defaults to function name)
            
        Returns:
            Decorated function
        """
        return self.monitor.track_function("api", operation)
    
    def track_service_function(self, operation: Optional[str] = None) -> Callable[[F], F]:
        """
        Decorator for tracking a service function.
        
        Args:
            operation: Operation name (defaults to function name)
            
        Returns:
            Decorated function
        """
        return self.monitor.track_function("service", operation)
    
    def track_repository_function(self, operation: Optional[str] = None) -> Callable[[F], F]:
        """
        Decorator for tracking a repository function.
        
        Args:
            operation: Operation name (defaults to function name)
            
        Returns:
            Decorated function
        """
        return self.monitor.track_function("repository", operation)
    
    def track_client_function(self, operation: Optional[str] = None) -> Callable[[F], F]:
        """
        Decorator for tracking a client function.
        
        Args:
            operation: Operation name (defaults to function name)
            
        Returns:
            Decorated function
        """
        return self.monitor.track_function("client", operation)
    
    def track_processor_function(self, operation: Optional[str] = None) -> Callable[[F], F]:
        """
        Decorator for tracking a processor function.
        
        Args:
            operation: Operation name (defaults to function name)
            
        Returns:
            Decorated function
        """
        return self.monitor.track_function("processor", operation)
    
    def track_validation_function(self, operation: Optional[str] = None) -> Callable[[F], F]:
        """
        Decorator for tracking a validation function.
        
        Args:
            operation: Operation name (defaults to function name)
            
        Returns:
            Decorated function
        """
        return self.monitor.track_function("validation", operation)
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """
        Get service-specific metrics.
        
        Returns:
            Service-specific metrics
        """
        metrics = {}
        
        # Get metrics for each component
        for component in COMPONENT_NAMES.keys():
            component_metrics = self.monitor.get_operation_metrics(component)
            if component_metrics:
                metrics[component] = component_metrics
        
        return metrics
    
    def get_service_performance_report(self) -> Dict[str, Any]:
        """
        Get service performance report.
        
        Returns:
            Service performance report
        """
        return {
            "service": SERVICE_NAME,
            "components": {
                component: {
                    "name": name,
                    "metrics": self.monitor.get_operation_metrics(component)
                }
                for component, name in COMPONENT_NAMES.items()
            },
            "resources": self.monitor.get_resource_metrics()
        }


# Create singleton instance
service_performance_monitor = ServicePerformanceMonitor.get_instance()


def get_service_performance_monitor() -> ServicePerformanceMonitor:
    """
    Get the singleton instance of the service performance monitor.
    
    Returns:
        ServicePerformanceMonitor instance
    """
    return service_performance_monitor


# API operation tracking
track_api_operation = service_performance_monitor.track_api_function
track_api = track_api_operation  # Alias


# Service operation tracking
track_service_operation = service_performance_monitor.track_service_function
track_service = track_service_operation  # Alias


# Repository operation tracking
track_repository_operation = service_performance_monitor.track_repository_function
track_repository = track_repository_operation  # Alias


# Client operation tracking
track_client_operation = service_performance_monitor.track_client_function
track_client = track_client_operation  # Alias


# Processor operation tracking
track_processor_operation = service_performance_monitor.track_processor_function
track_processor = track_processor_operation  # Alias


# Validation operation tracking
track_validation_operation = service_performance_monitor.track_validation_function
track_validation = track_validation_operation  # Alias


# Context managers
api_operation = service_performance_monitor.track_api_operation
service_operation = service_performance_monitor.track_service_operation
repository_operation = service_performance_monitor.track_repository_operation
client_operation = service_performance_monitor.track_client_operation
processor_operation = service_performance_monitor.track_processor_operation
validation_operation = service_performance_monitor.track_validation_operation
