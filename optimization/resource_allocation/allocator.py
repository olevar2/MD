"""
Resource Allocator

This module provides the main ResourceAllocator class for dynamically managing
and allocating system resources.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Type
from datetime import datetime
import json
import os
import time

from optimization.resource_allocation.models import (
    ResourceType, ServicePriority, ResourcePolicy,
    ServiceResourceConfig, ResourceUtilization,
    ScalingDecision, ResourceAllocationResult
)
from optimization.resource_allocation.controllers.interface import ResourceControllerInterface
from optimization.resource_allocation.controllers.kubernetes import KubernetesResourceController
from optimization.resource_allocation.controllers.docker import DockerResourceController
from optimization.resource_allocation.metrics.interface import MetricsProviderInterface
from optimization.resource_allocation.metrics.prometheus import PrometheusMetricsProvider
from optimization.resource_allocation.policies.fixed import FixedAllocationPolicy
from optimization.resource_allocation.policies.dynamic import DynamicAllocationPolicy
from optimization.resource_allocation.policies.priority import PriorityBasedAllocationPolicy
from optimization.resource_allocation.policies.adaptive import AdaptiveAllocationPolicy
from optimization.resource_allocation.policies.elastic import ElasticAllocationPolicy
from optimization.resource_allocation.utils.config import load_config, save_config
from optimization.resource_allocation.utils.monitoring import setup_monitoring

# Set up logging
logger = logging.getLogger(__name__)


class ResourceAllocator:
    """
    Main class for resource allocation.
    
    This class coordinates the allocation of resources to services based on
    their configuration, priority, and current utilization. It uses controllers
    to interact with the underlying infrastructure and metrics providers to
    monitor resource usage.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ResourceAllocator.
        
        Args:
            config: Optional configuration dictionary with parameters:
                - controller_type: Type of resource controller to use ('kubernetes', 'docker')
                - metrics_provider: Type of metrics provider to use ('prometheus')
                - default_policy: Default allocation policy to use ('fixed', 'dynamic', 'priority', 'adaptive', 'elastic')
                - service_configs: Dictionary mapping service names to their configurations
                - monitoring_interval: Interval for monitoring resource usage (in seconds)
                - allocation_interval: Interval for resource allocation (in seconds)
                - controller_config: Configuration for the resource controller
                - metrics_config: Configuration for the metrics provider
                - policy_config: Configuration for allocation policies
        """
        self.config = config or {}
        
        # Initialize controller
        controller_type = self.config.get('controller_type', 'kubernetes')
        controller_config = self.config.get('controller_config', {})
        self.controller = self._create_controller(controller_type, controller_config)
        
        # Initialize metrics provider
        metrics_provider = self.config.get('metrics_provider', 'prometheus')
        metrics_config = self.config.get('metrics_config', {})
        self.metrics = self._create_metrics_provider(metrics_provider, metrics_config)
        
        # Initialize policies
        policy_config = self.config.get('policy_config', {})
        self.policies = self._create_policies(policy_config)
        
        # Load service configurations
        service_configs = self.config.get('service_configs', {})
        self.service_configs = {}
        for service_name, config_data in service_configs.items():
            if isinstance(config_data, dict):
                self.service_configs[service_name] = ServiceResourceConfig.from_dict(config_data)
            else:
                self.service_configs[service_name] = config_data
        
        # Set up monitoring
        monitoring_interval = self.config.get('monitoring_interval', 60)  # seconds
        self.monitoring = setup_monitoring(self.metrics, monitoring_interval)
        
        # Set default policy
        self.default_policy_type = self.config.get('default_policy', 'dynamic')
        
        # Initialize state
        self.last_allocation = {}
        self.allocation_history = []
        
        logger.info(f"ResourceAllocator initialized with controller: {controller_type}, "
                   f"metrics provider: {metrics_provider}, "
                   f"default policy: {self.default_policy_type}")
    
    def _create_controller(self, controller_type: str, config: Dict[str, Any]) -> ResourceControllerInterface:
        """
        Create a resource controller based on the specified type.
        
        Args:
            controller_type: Type of controller to create ('kubernetes', 'docker')
            config: Configuration for the controller
            
        Returns:
            ResourceControllerInterface: The created controller
            
        Raises:
            ValueError: If the controller type is not supported
        """
        if controller_type.lower() == 'kubernetes':
            return KubernetesResourceController(config)
        elif controller_type.lower() == 'docker':
            return DockerResourceController(config)
        else:
            raise ValueError(f"Unsupported controller type: {controller_type}")
    
    def _create_metrics_provider(self, provider_type: str, config: Dict[str, Any]) -> MetricsProviderInterface:
        """
        Create a metrics provider based on the specified type.
        
        Args:
            provider_type: Type of metrics provider to create ('prometheus')
            config: Configuration for the metrics provider
            
        Returns:
            MetricsProviderInterface: The created metrics provider
            
        Raises:
            ValueError: If the provider type is not supported
        """
        if provider_type.lower() == 'prometheus':
            return PrometheusMetricsProvider(config)
        else:
            raise ValueError(f"Unsupported metrics provider type: {provider_type}")
    
    def _create_policies(self, config: Dict[str, Any]) -> Dict[ResourcePolicy, Any]:
        """
        Create allocation policies.
        
        Args:
            config: Configuration for the policies
            
        Returns:
            Dict[ResourcePolicy, Any]: Dictionary mapping policy types to policy instances
        """
        return {
            ResourcePolicy.FIXED: FixedAllocationPolicy(config.get('fixed', {})),
            ResourcePolicy.DYNAMIC: DynamicAllocationPolicy(config.get('dynamic', {})),
            ResourcePolicy.PRIORITY_BASED: PriorityBasedAllocationPolicy(config.get('priority', {})),
            ResourcePolicy.ADAPTIVE: AdaptiveAllocationPolicy(config.get('adaptive', {})),
            ResourcePolicy.ELASTIC: ElasticAllocationPolicy(config.get('elastic', {}))
        }
    
    def register_service(self, config: ServiceResourceConfig) -> bool:
        """
        Register a service for resource allocation.
        
        Args:
            config: Service resource configuration
            
        Returns:
            bool: True if the service was registered successfully, False otherwise
        """
        try:
            self.service_configs[config.service_name] = config
            logger.info(f"Registered service: {config.service_name}")
            return True
        except Exception as e:
            logger.error(f"Error registering service {config.service_name}: {str(e)}")
            return False
    
    def unregister_service(self, service_name: str) -> bool:
        """
        Unregister a service from resource allocation.
        
        Args:
            service_name: Name of the service to unregister
            
        Returns:
            bool: True if the service was unregistered successfully, False otherwise
        """
        try:
            if service_name in self.service_configs:
                del self.service_configs[service_name]
                logger.info(f"Unregistered service: {service_name}")
                return True
            else:
                logger.warning(f"Service not found: {service_name}")
                return False
        except Exception as e:
            logger.error(f"Error unregistering service {service_name}: {str(e)}")
            return False
    
    def get_service_config(self, service_name: str) -> Optional[ServiceResourceConfig]:
        """
        Get the configuration for a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Optional[ServiceResourceConfig]: Service configuration, or None if not found
        """
        return self.service_configs.get(service_name)
    
    def update_service_config(self, config: ServiceResourceConfig) -> bool:
        """
        Update the configuration for a service.
        
        Args:
            config: New service configuration
            
        Returns:
            bool: True if the service configuration was updated successfully, False otherwise
        """
        try:
            if config.service_name in self.service_configs:
                self.service_configs[config.service_name] = config
                logger.info(f"Updated service configuration: {config.service_name}")
                return True
            else:
                logger.warning(f"Service not found: {config.service_name}")
                return False
        except Exception as e:
            logger.error(f"Error updating service configuration {config.service_name}: {str(e)}")
            return False
    
    def allocate(self, service_name: str, resources: Optional[Dict[ResourceType, Dict[str, Any]]] = None) -> ResourceAllocationResult:
        """
        Allocate resources to a service.
        
        Args:
            service_name: Name of the service
            resources: Optional resources to allocate. If not provided, the service configuration will be used.
            
        Returns:
            ResourceAllocationResult: Result of the allocation
        """
        try:
            # Get service configuration
            service_config = self.get_service_config(service_name)
            if not service_config:
                return ResourceAllocationResult(
                    service_name=service_name,
                    timestamp=datetime.now(),
                    success=False,
                    resources={},
                    error=f"Service not registered: {service_name}"
                )
            
            # Use provided resources or get from configuration
            allocation_resources = resources or service_config.resources
            
            # Get current utilization
            utilization = self.metrics.get_service_utilization(service_name)
            
            # Get policy
            policy_type = service_config.policy
            policy = self.policies.get(policy_type)
            if not policy:
                policy = self.policies.get(ResourcePolicy[self.default_policy_type.upper()])
            
            # Calculate allocation
            decisions = policy.calculate_allocation(
                service_config=service_config,
                current_utilization=utilization,
                resources=allocation_resources
            )
            
            # Apply allocation
            result = self.controller.allocate_resources(
                service_name=service_name,
                resources=allocation_resources,
                decisions=decisions
            )
            
            # Record allocation
            self.last_allocation[service_name] = {
                'timestamp': datetime.now(),
                'resources': allocation_resources,
                'decisions': decisions,
                'result': result
            }
            
            # Add to history
            self.allocation_history.append({
                'service_name': service_name,
                'timestamp': datetime.now().isoformat(),
                'resources': {r.name: v for r, v in allocation_resources.items()},
                'decisions': [d.to_dict() for d in decisions],
                'success': result.success
            })
            
            # Trim history if needed
            max_history = self.config.get('max_history', 1000)
            if len(self.allocation_history) > max_history:
                self.allocation_history = self.allocation_history[-max_history:]
            
            return result
        except Exception as e:
            logger.error(f"Error allocating resources for service {service_name}: {str(e)}")
            return ResourceAllocationResult(
                service_name=service_name,
                timestamp=datetime.now(),
                success=False,
                resources={},
                error=str(e)
            )
    
    def allocate_all(self) -> Dict[str, ResourceAllocationResult]:
        """
        Allocate resources to all registered services.
        
        Returns:
            Dict[str, ResourceAllocationResult]: Dictionary mapping service names to allocation results
        """
        results = {}
        for service_name in self.service_configs:
            results[service_name] = self.allocate(service_name)
        return results
    
    def get_utilization(self, service_name: str) -> Optional[ResourceUtilization]:
        """
        Get the current resource utilization for a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Optional[ResourceUtilization]: Current utilization, or None if not available
        """
        try:
            return self.metrics.get_service_utilization(service_name)
        except Exception as e:
            logger.error(f"Error getting utilization for service {service_name}: {str(e)}")
            return None
    
    def get_all_utilization(self) -> Dict[str, ResourceUtilization]:
        """
        Get the current resource utilization for all services.
        
        Returns:
            Dict[str, ResourceUtilization]: Dictionary mapping service names to utilization
        """
        results = {}
        for service_name in self.service_configs:
            utilization = self.get_utilization(service_name)
            if utilization:
                results[service_name] = utilization
        return results
    
    def get_allocation_history(self, service_name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get the allocation history for a service or all services.
        
        Args:
            service_name: Optional name of the service. If not provided, history for all services is returned.
            limit: Maximum number of history entries to return
            
        Returns:
            List[Dict[str, Any]]: List of allocation history entries
        """
        if service_name:
            # Filter history for the specified service
            history = [entry for entry in self.allocation_history if entry['service_name'] == service_name]
        else:
            # Return history for all services
            history = self.allocation_history
        
        # Sort by timestamp (newest first) and limit
        sorted_history = sorted(history, key=lambda x: x['timestamp'], reverse=True)
        return sorted_history[:limit]
    
    def save_state(self, file_path: str) -> bool:
        """
        Save the current state to a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if the state was saved successfully, False otherwise
        """
        try:
            state = {
                'service_configs': {name: config.to_dict() for name, config in self.service_configs.items()},
                'last_allocation': {
                    name: {
                        'timestamp': data['timestamp'].isoformat(),
                        'resources': {r.name: v for r, v in data['resources'].items()},
                        'decisions': [d.to_dict() for d in data['decisions']],
                        'result': data['result'].to_dict()
                    }
                    for name, data in self.last_allocation.items()
                },
                'allocation_history': self.allocation_history
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"State saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving state to {file_path}: {str(e)}")
            return False
    
    def load_state(self, file_path: str) -> bool:
        """
        Load state from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if the state was loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"State file not found: {file_path}")
                return False
            
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Load service configurations
            self.service_configs = {}
            for name, config_data in state.get('service_configs', {}).items():
                self.service_configs[name] = ServiceResourceConfig.from_dict(config_data)
            
            # Load allocation history
            self.allocation_history = state.get('allocation_history', [])
            
            # Load last allocation (partially, as we can't fully reconstruct the objects)
            self.last_allocation = {}
            for name, data in state.get('last_allocation', {}).items():
                self.last_allocation[name] = {
                    'timestamp': datetime.fromisoformat(data['timestamp']),
                    'resources': {ResourceType[r]: v for r, v in data['resources'].items()},
                    'decisions': [ScalingDecision.from_dict(d) for d in data['decisions']],
                    'result': ResourceAllocationResult.from_dict(data['result'])
                }
            
            logger.info(f"State loaded from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading state from {file_path}: {str(e)}")
            return False