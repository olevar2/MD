"""
Service Registry and Discovery

This module provides a centralized registry for service discovery, health monitoring,
and automatic failover between redundant service instances.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import json
import uuid
import random

from common_lib.exceptions import ServiceUnavailableError
from core_foundations.monitoring.health_check import HealthCheck, HealthStatus
from core_foundations.resilience.degraded_mode import DegradedModeManager

logger = logging.getLogger(__name__)


class ServiceState(str, Enum):
    """Service instance state"""
    HEALTHY = "HEALTHY"           # Service is healthy and available
    DEGRADED = "DEGRADED"         # Service is available but operating in degraded mode
    UNHEALTHY = "UNHEALTHY"       # Service is available but health checks are failing
    UNAVAILABLE = "UNAVAILABLE"   # Service is not responding
    MAINTENANCE = "MAINTENANCE"   # Service is temporarily unavailable for maintenance
    STARTING = "STARTING"         # Service is starting up
    STOPPING = "STOPPING"         # Service is shutting down
    UNKNOWN = "UNKNOWN"           # Service state is unknown


class ServiceInstance:
    """
    Represents an instance of a service in the registry.
    """
    
    def __init__(
        self,
        instance_id: str,
        service_name: str,
        host: str,
        port: int,
        metadata: Dict[str, Any] = None,
        health_endpoint: str = "/health",
        weight: int = 100,
        priority: int = 1
    ):
        """
        Initialize a service instance.
        
        Args:
            instance_id: Unique identifier for this instance
            service_name: Name of the service
            host: Host address (IP or hostname)
            port: Port number
            metadata: Additional metadata about the instance
            health_endpoint: URI path for health checks
            weight: Relative weight for load balancing (1-100)
            priority: Priority for failover (lower is higher priority)
        """
        self.instance_id = instance_id
        self.service_name = service_name
        self.host = host
        self.port = port
        self.metadata = metadata or {}
        self.health_endpoint = health_endpoint
        self.weight = max(1, min(100, weight))  # Ensure weight is between 1 and 100
        self.priority = max(1, priority)  # Priority must be at least 1
        
        self.registration_time = datetime.utcnow()
        self.last_heartbeat = datetime.utcnow()
        self.state = ServiceState.STARTING
        self.state_reason = "Initial registration"
        self.state_last_updated = datetime.utcnow()
        
        # Health metrics
        self.successful_requests = 0
        self.failed_requests = 0
        self.average_response_time = 0.0
        self.last_health_check = None
        
        # Degraded mode information
        self.degraded_dependencies = set()
    
    def get_url(self, path: str = "") -> str:
        """
        Get the URL for this service instance.
        
        Args:
            path: Optional path to append to the URL
            
        Returns:
            Full URL to the service instance
        """
        base_url = f"http://{self.host}:{self.port}"
        if path:
            if not path.startswith("/"):
                path = f"/{path}"
            return f"{base_url}{path}"
        return base_url
    
    def get_health_url(self) -> str:
        """Get the health check URL for this instance"""
        return self.get_url(self.health_endpoint)
    
    def update_heartbeat(self) -> None:
        """Update the last heartbeat time"""
        self.last_heartbeat = datetime.utcnow()
    
    def update_state(self, state: ServiceState, reason: str = "") -> None:
        """
        Update the instance state.
        
        Args:
            state: New state
            reason: Reason for the state change
        """
        if self.state != state:
            old_state = self.state
            self.state = state
            self.state_reason = reason
            self.state_last_updated = datetime.utcnow()
            logger.info(f"Service {self.service_name} ({self.instance_id}) state changed: {old_state} -> {state} ({reason})")
    
    def update_health_metrics(
        self,
        success: bool,
        response_time: float,
        health_data: Dict[str, Any] = None
    ) -> None:
        """
        Update health metrics for this instance.
        
        Args:
            success: Whether the health check was successful
            response_time: Response time in milliseconds
            health_data: Health check response data
        """
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Update average response time with exponential smoothing
        alpha = 0.3  # Smoothing factor
        if self.average_response_time == 0.0:
            self.average_response_time = response_time
        else:
            self.average_response_time = (alpha * response_time) + ((1 - alpha) * self.average_response_time)
        
        self.last_health_check = health_data
    
    def is_healthy(self) -> bool:
        """Check if the instance is considered healthy"""
        return self.state in (ServiceState.HEALTHY, ServiceState.DEGRADED)
    
    def is_available(self) -> bool:
        """Check if the instance is available for requests"""
        return self.state in (ServiceState.HEALTHY, ServiceState.DEGRADED)
    
    def update_degraded_dependencies(self, dependencies: Set[str]) -> None:
        """
        Update the degraded dependencies for this instance.
        
        Args:
            dependencies: Set of degraded dependency names
        """
        old_dependencies = self.degraded_dependencies.copy()
        self.degraded_dependencies = dependencies
        
        # Update state based on dependencies
        if dependencies and not old_dependencies:
            self.update_state(ServiceState.DEGRADED, f"Dependencies in degraded mode: {', '.join(dependencies)}")
        elif not dependencies and old_dependencies:
            self.update_state(ServiceState.HEALTHY, "No dependencies in degraded mode")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the instance to a dictionary.
        
        Returns:
            Dictionary representation of the instance
        """
        return {
            "instance_id": self.instance_id,
            "service_name": self.service_name,
            "host": self.host,
            "port": self.port,
            "url": self.get_url(),
            "health_url": self.get_health_url(),
            "metadata": self.metadata,
            "weight": self.weight,
            "priority": self.priority,
            "registration_time": self.registration_time.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "state": self.state.value,
            "state_reason": self.state_reason,
            "state_last_updated": self.state_last_updated.isoformat(),
            "metrics": {
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "average_response_time": self.average_response_time,
            },
            "degraded_dependencies": list(self.degraded_dependencies)
        }


class ServiceSelectorStrategy(str, Enum):
    """Strategy for selecting service instances"""
    ROUND_ROBIN = "ROUND_ROBIN"       # Select instances in rotation
    RANDOM = "RANDOM"                 # Select instances randomly
    LEAST_CONNECTIONS = "LEAST_CONNECTIONS"  # Select instance with least active connections
    LOWEST_LATENCY = "LOWEST_LATENCY"  # Select instance with lowest response time
    WEIGHTED = "WEIGHTED"             # Select instances based on weight
    PRIORITY = "PRIORITY"             # Select highest priority instance(s)


class ServiceRegistry:
    """
    Centralized registry for service discovery and health monitoring.
    """
    
    _instance = None
    
    def __new__(cls):
        """Create a singleton instance"""
        if cls._instance is None:
            cls._instance = super(ServiceRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the service registry"""
        if self._initialized:
            return
        
        # Dictionary mapping service names to lists of instances
        self._services = {}
        
        # Dictionary mapping instance IDs to instances
        self._instances = {}
        
        # Last selected instance index for round-robin selection
        self._last_selected = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Initialize stopped flag
        self._stopped = False
        
        # Schedule background tasks
        self._schedule_tasks()
        
        self._initialized = True
    
    def _schedule_tasks(self) -> None:
        """Schedule background tasks"""
        # Health check thread
        self._health_check_thread = threading.Thread(
            target=self._health_check_task,
            daemon=True,
            name="service-registry-health"
        )
        self._health_check_thread.start()
        
        # Cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_task,
            daemon=True,
            name="service-registry-cleanup"
        )
        self._cleanup_thread.start()
    
    def register_service_instance(self, instance: ServiceInstance) -> None:
        """
        Register a service instance.
        
        Args:
            instance: Service instance to register
        """
        with self._lock:
            service_name = instance.service_name
            
            if service_name not in self._services:
                self._services[service_name] = []
            
            # Add to services list
            self._services[service_name].append(instance)
            
            # Add to instances map
            self._instances[instance.instance_id] = instance
            
            logger.info(f"Registered service instance: {service_name} ({instance.instance_id})")
    
    def unregister_service_instance(self, instance_id: str) -> bool:
        """
        Unregister a service instance.
        
        Args:
            instance_id: ID of instance to unregister
            
        Returns:
            True if instance was found and unregistered, False otherwise
        """
        with self._lock:
            if instance_id not in self._instances:
                logger.warning(f"Cannot unregister instance {instance_id}: not found")
                return False
            
            instance = self._instances[instance_id]
            service_name = instance.service_name
            
            # Remove from services list
            if service_name in self._services:
                self._services[service_name] = [
                    i for i in self._services[service_name] 
                    if i.instance_id != instance_id
                ]
                
                # Remove service if no instances left
                if not self._services[service_name]:
                    del self._services[service_name]
            
            # Remove from instances map
            del self._instances[instance_id]
            
            logger.info(f"Unregistered service instance: {service_name} ({instance_id})")
            return True
    
    def get_service_instances(
        self, 
        service_name: str, 
        include_unhealthy: bool = False
    ) -> List[ServiceInstance]:
        """
        Get all instances of a service.
        
        Args:
            service_name: Name of the service
            include_unhealthy: Whether to include unhealthy instances
            
        Returns:
            List of service instances
        """
        with self._lock:
            if service_name not in self._services:
                return []
            
            instances = self._services[service_name]
            
            if not include_unhealthy:
                instances = [i for i in instances if i.is_healthy()]
            
            return list(instances)  # Return a copy of the list
    
    def get_service_instance(
        self, 
        service_name: str, 
        strategy: ServiceSelectorStrategy = ServiceSelectorStrategy.PRIORITY,
        exclude_instance_ids: List[str] = None
    ) -> Optional[ServiceInstance]:
        """
        Get a service instance using the specified selection strategy.
        
        Args:
            service_name: Name of the service
            strategy: Strategy for selecting instances
            exclude_instance_ids: Instance IDs to exclude
            
        Returns:
            Selected service instance, or None if no instances available
        """
        with self._lock:
            # Get all healthy instances
            instances = self.get_service_instances(service_name)
            
            if not instances:
                return None
            
            # Filter out excluded instances
            if exclude_instance_ids:
                instances = [i for i in instances if i.instance_id not in exclude_instance_ids]
                
                if not instances:
                    return None
            
            # Apply selection strategy
            if strategy == ServiceSelectorStrategy.ROUND_ROBIN:
                # Get last selected index
                index = self._last_selected.get(service_name, -1)
                
                # Update index
                index = (index + 1) % len(instances)
                self._last_selected[service_name] = index
                
                return instances[index]
                
            elif strategy == ServiceSelectorStrategy.RANDOM:
                return random.choice(instances)
                
            elif strategy == ServiceSelectorStrategy.LOWEST_LATENCY:
                # Sort by average response time
                instances.sort(key=lambda i: i.average_response_time)
                return instances[0]
                
            elif strategy == ServiceSelectorStrategy.WEIGHTED:
                # Weighted random selection
                total_weight = sum(i.weight for i in instances)
                if total_weight <= 0:
                    return random.choice(instances)
                
                r = random.uniform(0, total_weight)
                cumulative_weight = 0
                
                for instance in instances:
                    cumulative_weight += instance.weight
                    if cumulative_weight >= r:
                        return instance
                
                # Should not reach here, but just in case
                return instances[-1]
                
            elif strategy == ServiceSelectorStrategy.PRIORITY:
                # Sort by priority (lower is higher priority)
                instances.sort(key=lambda i: i.priority)
                return instances[0]
                
            else:
                # Default to first instance
                return instances[0] if instances else None
    
    def get_all_services(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all services in the registry.
        
        Returns:
            Dictionary mapping service names to lists of instance dictionaries
        """
        with self._lock:
            result = {}
            
            for service_name, instances in self._services.items():
                result[service_name] = [
                    instance.to_dict() for instance in instances
                ]
            
            return result
    
    def update_heartbeat(self, instance_id: str) -> bool:
        """
        Update the heartbeat for a service instance.
        
        Args:
            instance_id: ID of the instance
            
        Returns:
            True if instance was found, False otherwise
        """
        with self._lock:
            if instance_id not in self._instances:
                return False
            
            self._instances[instance_id].update_heartbeat()
            return True
    
    def update_instance_state(
        self, 
        instance_id: str, 
        state: ServiceState, 
        reason: str = ""
    ) -> bool:
        """
        Update the state of a service instance.
        
        Args:
            instance_id: ID of the instance
            state: New state
            reason: Reason for the state change
            
        Returns:
            True if instance was found, False otherwise
        """
        with self._lock:
            if instance_id not in self._instances:
                return False
            
            self._instances[instance_id].update_state(state, reason)
            return True
    
    def _health_check_task(self) -> None:
        """Background task for checking instance health"""
        health_check_interval = 15.0  # seconds
        
        while not self._stopped:
            try:
                # Sleep first to give time for instances to initialize
                time.sleep(health_check_interval)
                
                now = datetime.utcnow()
                heartbeat_threshold = now - timedelta(seconds=30)
                
                with self._lock:
                    for instance_id, instance in list(self._instances.items()):
                        # Check heartbeat
                        if instance.last_heartbeat < heartbeat_threshold:
                            # Instance may be down
                            if instance.state != ServiceState.UNAVAILABLE:
                                instance.update_state(
                                    ServiceState.UNAVAILABLE, 
                                    f"No heartbeat since {instance.last_heartbeat.isoformat()}"
                                )
            
            except Exception as e:
                logger.error(f"Error in health check task: {e}")
    
    def _cleanup_task(self) -> None:
        """Background task for cleaning up stale instances"""
        cleanup_interval = 60.0  # seconds
        
        while not self._stopped:
            try:
                # Sleep first
                time.sleep(cleanup_interval)
                
                now = datetime.utcnow()
                stale_threshold = now - timedelta(minutes=5)
                
                with self._lock:
                    for instance_id, instance in list(self._instances.items()):
                        # Check if instance is stale
                        if (instance.state == ServiceState.UNAVAILABLE and 
                            instance.last_heartbeat < stale_threshold):
                            
                            # Unregister stale instance
                            self.unregister_service_instance(instance_id)
                            logger.info(f"Removed stale service instance: {instance.service_name} ({instance_id})")
            
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    def stop(self) -> None:
        """Stop background tasks"""
        self._stopped = True


# Client for interacting with service registry
class ServiceRegistryClient:
    """
    Client for interacting with a service registry.
    
    This class provides methods for service registration, heartbeat, and discovery.
    """
    
    def __init__(
        self,
        service_name: str,
        host: str,
        port: int,
        metadata: Dict[str, Any] = None,
        instance_id: str = None
    ):
        """
        Initialize the service registry client.
        
        Args:
            service_name: Name of the service
            host: Host address (IP or hostname)
            port: Port number
            metadata: Additional metadata about the instance
            instance_id: Unique identifier for this instance (auto-generated if None)
        """
        self.service_name = service_name
        self.host = host
        self.port = port
        self.metadata = metadata or {}
        
        # Generate instance ID if not provided
        self.instance_id = instance_id or f"{service_name}-{uuid.uuid4().hex[:8]}"
        
        # Initialize registry and instance
        self._registry = ServiceRegistry()
        self._instance = ServiceInstance(
            instance_id=self.instance_id,
            service_name=self.service_name,
            host=self.host,
            port=self.port,
            metadata=self.metadata
        )
        
        # Initialize degraded mode manager
        self._degraded_manager = DegradedModeManager()
        
        # Initialize heartbeat thread
        self._stop_heartbeat = False
        self._heartbeat_thread = None
    
    def register(self) -> None:
        """Register this service instance"""
        # Register instance
        self._registry.register_service_instance(self._instance)
        
        # Set up degraded mode handler
        self._degraded_manager.add_degraded_mode_handler(self._handle_degraded_mode)
        
        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_task,
            daemon=True,
            name="service-registry-heartbeat"
        )
        self._heartbeat_thread.start()
    
    def unregister(self) -> None:
        """Unregister this service instance"""
        # Stop heartbeat thread
        self._stop_heartbeat = True
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=1.0)
        
        # Unregister instance
        self._registry.unregister_service_instance(self.instance_id)
        
        # Remove degraded mode handler
        self._degraded_manager.remove_degraded_mode_handler(self._handle_degraded_mode)
    
    def _heartbeat_task(self) -> None:
        """Background task for sending heartbeats"""
        heartbeat_interval = 10.0  # seconds
        
        while not self._stop_heartbeat:
            try:
                # Send heartbeat
                self._registry.update_heartbeat(self.instance_id)
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
            
            # Sleep
            time.sleep(heartbeat_interval)
    
    def update_state(self, state: ServiceState, reason: str = "") -> None:
        """
        Update the state of this service instance.
        
        Args:
            state: New state
            reason: Reason for the state change
        """
        self._registry.update_instance_state(self.instance_id, state, reason)
        
        # Also update local instance
        self._instance.update_state(state, reason)
    
    def _handle_degraded_mode(self, dependency_name: str, is_degraded: bool) -> None:
        """
        Handle changes in degraded mode status.
        
        Args:
            dependency_name: Name of the dependency
            is_degraded: Whether the dependency is in degraded mode
        """
        # Update degraded dependencies
        if is_degraded:
            self._instance.degraded_dependencies.add(dependency_name)
        else:
            self._instance.degraded_dependencies.discard(dependency_name)
        
        # Update instance state
        if self._instance.degraded_dependencies:
            self._instance.update_state(
                ServiceState.DEGRADED, 
                f"Dependencies in degraded mode: {', '.join(self._instance.degraded_dependencies)}"
            )
        else:
            self._instance.update_state(ServiceState.HEALTHY, "No dependencies in degraded mode")
            
        # Update in registry
        self._registry.update_instance_state(
            self.instance_id, 
            self._instance.state, 
            self._instance.state_reason
        )
    
    def get_service_url(
        self, 
        service_name: str, 
        path: str = "",
        strategy: ServiceSelectorStrategy = ServiceSelectorStrategy.PRIORITY,
        exclude_instance_ids: List[str] = None
    ) -> str:
        """
        Get the URL for a service.
        
        Args:
            service_name: Name of the service
            path: Optional path to append to the URL
            strategy: Strategy for selecting instances
            exclude_instance_ids: Instance IDs to exclude
            
        Returns:
            URL to the service
            
        Raises:
            ServiceUnavailableError: If no instances available
        """
        instance = self._registry.get_service_instance(
            service_name, 
            strategy=strategy,
            exclude_instance_ids=exclude_instance_ids
        )
        
        if not instance:
            raise ServiceUnavailableError(f"No available instances of service: {service_name}")
        
        if path:
            return instance.get_url(path)
        
        return instance.get_url()
    
    def get_service_instances(
        self, 
        service_name: str, 
        include_unhealthy: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all instances of a service.
        
        Args:
            service_name: Name of the service
            include_unhealthy: Whether to include unhealthy instances
            
        Returns:
            List of service instance dictionaries
        """
        instances = self._registry.get_service_instances(
            service_name, 
            include_unhealthy=include_unhealthy
        )
        
        return [instance.to_dict() for instance in instances]


def create_service_registry_client(
    service_name: str,
    host: str,
    port: int,
    metadata: Dict[str, Any] = None,
    auto_register: bool = True
) -> ServiceRegistryClient:
    """
    Create a service registry client.
    
    Args:
        service_name: Name of the service
        host: Host address (IP or hostname)
        port: Port number
        metadata: Additional metadata about the instance
        auto_register: Whether to automatically register the service
        
    Returns:
        ServiceRegistryClient instance
    """
    client = ServiceRegistryClient(
        service_name=service_name,
        host=host,
        port=port,
        metadata=metadata
    )
    
    if auto_register:
        client.register()
    
    return client


def integrate_health_check_with_registry(
    health_check: HealthCheck, 
    registry_client: ServiceRegistryClient
) -> None:
    """
    Integrate a health check with the service registry.
    
    Updates the service state based on health check results.
    
    Args:
        health_check: Health check instance
        registry_client: Service registry client
    """
    def health_check_listener(health):
        """Update service state based on health check results"""
        if health.status == HealthStatus.UP:
            registry_client.update_state(ServiceState.HEALTHY, "Health check passing")
        elif health.status == HealthStatus.PARTIAL:
            registry_client.update_state(ServiceState.DEGRADED, "Health check partial")
        else:
            registry_client.update_state(ServiceState.UNHEALTHY, "Health check failing")
    
    health_check.add_listener(health_check_listener)
