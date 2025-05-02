"""
Resource Allocator Module

This module implements a ResourceAllocator class for dynamically managing and allocating system resources
(CPU, memory) based on priority, load, and predefined policies. Includes adaptive scaling logic and
resource usage optimization algorithms.

The ResourceAllocator is designed to be used by service orchestration components (like Kubernetes controllers
or custom schedulers) to efficiently distribute resources across the trading platform components.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Set
import logging
import time
import threading
import json
import os
import traceback
from datetime import datetime, timedelta
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Import error handling
try:
    from optimization.error import (
        OptimizationError,
        ResourceAllocationError,
        ParameterValidationError,
        with_error_handling
    )
except ImportError:
    # Define placeholder for error handling if module not available
    def with_error_handling(error_class=Exception, reraise=True, cleanup_func=None):
        def decorator(func):
            return func
        return decorator

    class OptimizationError(Exception):
        pass

    class ResourceAllocationError(OptimizationError):
        pass

    class ParameterValidationError(OptimizationError):
        pass

# Try importing core_foundations if available, otherwise use standard logging
try:
    from core_foundations.utils.logger import get_logger
    logger = get_logger("resource-allocator")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("resource-allocator")


class ResourceType(str, Enum):
    """Types of resources that can be allocated."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    GPU = "gpu"


class ServicePriority(int, Enum):
    """Priority levels for services."""
    CRITICAL = 5    # Trading gateway, risk management
    HIGH = 4        # Strategy execution, portfolio management
    MEDIUM = 3      # Analysis engine, feature store
    LOW = 2         # Batch processes, backtest jobs
    BACKGROUND = 1  # Monitoring, non-critical utilities


class ResourcePolicy(str, Enum):
    """Resource allocation policy types."""
    FIXED = "fixed"              # Fixed allocation regardless of usage
    DYNAMIC = "dynamic"          # Allocation changes based on actual usage
    PRIORITY_BASED = "priority"  # Higher priority services get resources first
    ADAPTIVE = "adaptive"        # Uses ML to predict and allocate resources
    ELASTIC = "elastic"          # Quick response to resource demands


@dataclass
class ServiceResourceConfig:
    """Configuration for service resource allocation."""
    name: str
    priority: ServicePriority
    policy: ResourcePolicy
    min_resources: Dict[ResourceType, float]
    max_resources: Dict[ResourceType, float]
    target_resources: Dict[ResourceType, float] = field(default_factory=dict)
    scaling_factor: float = 1.0
    cooldown_seconds: int = 60
    last_scaling_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceUtilization:
    """Represents current resource utilization."""
    service_name: str
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_io_percent: float = 0.0
    network_io_percent: float = 0.0
    gpu_percent: float = 0.0
    queue_length: Optional[int] = None  # Optional: Number of items in service queue
    request_latency_ms: Optional[float] = None # Optional: Average request latency in milliseconds
    timestamp: datetime = field(default_factory=datetime.now)

    def get_utilization(self, resource_type: ResourceType) -> float:
        """Get the utilization percentage for a specific resource type."""
        if resource_type == ResourceType.CPU:
            return self.cpu_percent
        elif resource_type == ResourceType.MEMORY:
            return self.memory_percent
        elif resource_type == ResourceType.DISK_IO:
            return self.disk_io_percent
        elif resource_type == ResourceType.NETWORK_IO:
            return self.network_io_percent
        elif resource_type == ResourceType.GPU:
            return self.gpu_percent
        return 0.0


class ResourceAllocator:
    """
    Class for dynamically managing and allocating system resources based on
    priority, load, and predefined policies.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        metrics_provider: Optional[Any] = None,
        resource_controller: Optional[Any] = None,
        update_interval_seconds: int = 30,
        enable_predictive_scaling: bool = False,
        enable_load_balancing: bool = True,
    ):
        """
        Initialize the ResourceAllocator.

        Args:
            config_path: Path to configuration file (optional)
            metrics_provider: Provider for system metrics (optional)
            resource_controller: Controller for applying resource changes (optional)
            update_interval_seconds: Interval for resource allocation updates
            enable_predictive_scaling: Whether to use predictive scaling
            enable_load_balancing: Whether to enable load balancing
        """
        self.services: Dict[str, ServiceResourceConfig] = {}
        self.utilization_history: Dict[str, List[ResourceUtilization]] = {}
        self.update_interval = update_interval_seconds
        self.enable_predictive_scaling = enable_predictive_scaling
        self.enable_load_balancing = enable_load_balancing
        self.available_resources = {
            ResourceType.CPU: 100.0,  # Total percentage across all cores
            ResourceType.MEMORY: 100.0,  # Total percentage of system memory
            ResourceType.DISK_IO: 100.0,  # Percentage of max IO capacity
            ResourceType.NETWORK_IO: 100.0,  # Percentage of network capacity
            ResourceType.GPU: 100.0 if self._has_gpu() else 0.0  # GPU if available
        }

        self._resource_lock = threading.RLock()
        self._running = False
        self._allocation_thread = None
        self._metrics_provider = metrics_provider
        self._resource_controller = resource_controller
        self._market_regime = "normal"

        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

        # Historical scaling decisions for feedback learning
        self.scaling_history: List[Dict[str, Any]] = []

        # Resource reservation for critical spikes
        self.resource_buffer = {
            ResourceType.CPU: 10.0,  # Reserve 10% CPU for spikes
            ResourceType.MEMORY: 15.0,  # Reserve 15% memory
            ResourceType.DISK_IO: 20.0,
            ResourceType.NETWORK_IO: 20.0,
            ResourceType.GPU: 5.0 if self._has_gpu() else 0.0
        }

        # Adaptivity settings based on market conditions
        self.market_scaling_factors = {
            "volatile": {
                ResourceType.CPU: 1.3,  # More CPU during volatile markets
                ResourceType.MEMORY: 1.2,
                ResourceType.DISK_IO: 1.1,
                ResourceType.NETWORK_IO: 1.3,
                ResourceType.GPU: 1.2,
            },
            "trending": {
                ResourceType.CPU: 1.1,
                ResourceType.MEMORY: 1.0,
                ResourceType.DISK_IO: 1.0,
                ResourceType.NETWORK_IO: 1.1,
                ResourceType.GPU: 1.0,
            },
            "ranging": {
                ResourceType.CPU: 0.9,
                ResourceType.MEMORY: 0.9,
                ResourceType.DISK_IO: 0.9,
                ResourceType.NETWORK_IO: 0.9,
                ResourceType.GPU: 0.8,
            },
            "normal": {
                ResourceType.CPU: 1.0,
                ResourceType.MEMORY: 1.0,
                ResourceType.DISK_IO: 1.0,
                ResourceType.NETWORK_IO: 1.0,
                ResourceType.GPU: 1.0,
            },
            "after_hours": {
                ResourceType.CPU: 0.7,
                ResourceType.MEMORY: 0.8,
                ResourceType.DISK_IO: 1.2,  # More disk for batch jobs and backups
                ResourceType.NETWORK_IO: 0.7,
                ResourceType.GPU: 1.3,  # More GPU for training models overnight
            }
        }

        # Initialize optimization algorithms
        self.optimization_algorithms = {
            ResourcePolicy.FIXED: self._fixed_allocation,
            ResourcePolicy.DYNAMIC: self._dynamic_allocation,
            ResourcePolicy.PRIORITY_BASED: self._priority_based_allocation,
            ResourcePolicy.ADAPTIVE: self._adaptive_allocation,
            ResourcePolicy.ELASTIC: self._elastic_allocation
        }

        logger.info("ResourceAllocator initialized")

    def _has_gpu(self) -> bool:
        """Check if GPU is available."""
        # In a real implementation, this would detect actual GPU availability
        # This is a simplified placeholder
        return False

    @with_error_handling(error_class=ResourceAllocationError)
    def load_config(self, config_path: str) -> None:
        """
        Load resource allocation configuration from a JSON file.

        Args:
            config_path: Path to the configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            # Clear existing configuration
            self.services.clear()

            # Process service configurations
            self._process_service_configs(config_data.get("services", []))

            # Load available resources if specified
            self._load_resource_settings(config_data)

            logger.info(f"Loaded configuration from {config_path} with {len(self.services)} services")

        except Exception as e:
            error_details = {
                "config_path": config_path,
                "traceback": traceback.format_exc()
            }
            logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
            raise ResourceAllocationError(
                message=f"Failed to load configuration: {str(e)}",
                details=error_details
            ) from e

    @with_error_handling(error_class=ResourceAllocationError)
    def _process_service_configs(self, service_configs: List[Dict[str, Any]]) -> None:
        """
        Process service configurations from loaded config data.

        Args:
            service_configs: List of service configuration dictionaries
        """
        for service_config in service_configs:
            name = service_config.get("name")
            if not name:
                logger.warning("Skipping service config with missing name")
                continue

            try:
                priority = ServicePriority(int(service_config.get("priority", 3)))
            except (ValueError, TypeError):
                priority = ServicePriority.MEDIUM

            try:
                policy = ResourcePolicy(service_config.get("policy", "dynamic"))
            except ValueError:
                policy = ResourcePolicy.DYNAMIC

            min_resources = {
                ResourceType(k): float(v)
                for k, v in service_config.get("min_resources", {}).items()
                if k in [r.value for r in ResourceType]
            }

            max_resources = {
                ResourceType(k): float(v)
                for k, v in service_config.get("max_resources", {}).items()
                if k in [r.value for r in ResourceType]
            }

            target_resources = {
                ResourceType(k): float(v)
                for k, v in service_config.get("target_resources", {}).items()
                if k in [r.value for r in ResourceType]
            }

            scaling_factor = float(service_config.get("scaling_factor", 1.0))
            cooldown_seconds = int(service_config.get("cooldown_seconds", 60))

            service_resource_config = ServiceResourceConfig(
                name=name,
                priority=priority,
                policy=policy,
                min_resources=min_resources,
                max_resources=max_resources,
                target_resources=target_resources,
                scaling_factor=scaling_factor,
                cooldown_seconds=cooldown_seconds,
                metadata=service_config.get("metadata", {})
            )

            self.services[name] = service_resource_config

    @with_error_handling(error_class=ResourceAllocationError)
    def _load_resource_settings(self, config_data: Dict[str, Any]) -> None:
        """
        Load resource settings from config data.

        Args:
            config_data: Configuration data dictionary
        """
        # Load available resources if specified
        if "available_resources" in config_data:
            for resource_type_str, value in config_data["available_resources"].items():
                try:
                    resource_type = ResourceType(resource_type_str)
                    self.available_resources[resource_type] = float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid resource type or value: {resource_type_str}={value}")
                    continue

        # Load resource buffer if specified
        if "resource_buffer" in config_data:
            for resource_type_str, value in config_data["resource_buffer"].items():
                try:
                    resource_type = ResourceType(resource_type_str)
                    self.resource_buffer[resource_type] = float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid resource type or value: {resource_type_str}={value}")
                    continue

    def start(self) -> None:
        """Start the resource allocation service."""
        if self._running:
            logger.warning("Resource allocator is already running")
            return

        self._running = True
        self._allocation_thread = threading.Thread(
            target=self._resource_allocation_loop,
            daemon=True
        )
        self._allocation_thread.start()
        logger.info("Resource allocator started")

    def stop(self) -> None:
        """Stop the resource allocation service."""
        if not self._running:
            logger.warning("Resource allocator is already stopped")
            return

        self._running = False
        if self._allocation_thread:
            self._allocation_thread.join(timeout=5.0)
        logger.info("Resource allocator stopped")

    def register_service(self, service_config: ServiceResourceConfig) -> None:
        """
        Register a service for resource allocation.

        Args:
            service_config: Service resource configuration
        """
        with self._resource_lock:
            self.services[service_config.name] = service_config
            self.utilization_history[service_config.name] = []
            logger.info(f"Registered service: {service_config.name} with {service_config.policy.value} policy")

    def unregister_service(self, service_name: str) -> None:
        """
        Unregister a service from resource allocation.

        Args:
            service_name: Name of the service to unregister
        """
        with self._resource_lock:
            if service_name in self.services:
                self.services.pop(service_name)
                self.utilization_history.pop(service_name, None)
                logger.info(f"Unregistered service: {service_name}")
            else:
                logger.warning(f"Attempted to unregister unknown service: {service_name}")

    @with_error_handling(error_class=ParameterValidationError)
    def update_service_config(self, service_name: str, **kwargs) -> None:
        """
        Update configuration for a registered service.

        Args:
            service_name: Name of the service
            **kwargs: Configuration parameters to update
        """
        with self._resource_lock:
            if service_name not in self.services:
                raise ParameterValidationError(
                    message=f"Attempted to update unknown service: {service_name}",
                    parameter_name="service_name",
                    parameter_value=service_name
                )

            service = self.services[service_name]

            # Update fields by type
            self._update_service_priority(service, kwargs.get("priority"))
            self._update_service_policy(service, kwargs.get("policy"))
            self._update_service_scalar_params(service, kwargs)
            self._update_service_resources(service, kwargs)

            # Update metadata if provided
            if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
                service.metadata.update(kwargs["metadata"])

            logger.info(f"Updated configuration for service: {service_name}")

    @with_error_handling(error_class=ParameterValidationError)
    def _update_service_priority(self, service: ServiceResourceConfig, priority_value: Optional[int]) -> None:
        """
        Update service priority.

        Args:
            service: Service configuration to update
            priority_value: New priority value
        """
        if priority_value is not None and isinstance(priority_value, int):
            try:
                service.priority = ServicePriority(priority_value)
            except ValueError:
                raise ParameterValidationError(
                    message=f"Invalid priority value: {priority_value}",
                    parameter_name="priority",
                    parameter_value=priority_value
                )

    @with_error_handling(error_class=ParameterValidationError)
    def _update_service_policy(self, service: ServiceResourceConfig, policy_value: Optional[str]) -> None:
        """
        Update service policy.

        Args:
            service: Service configuration to update
            policy_value: New policy value
        """
        if policy_value is not None and isinstance(policy_value, str):
            try:
                service.policy = ResourcePolicy(policy_value)
            except ValueError:
                raise ParameterValidationError(
                    message=f"Invalid policy value: {policy_value}",
                    parameter_name="policy",
                    parameter_value=policy_value
                )

    @with_error_handling(error_class=ParameterValidationError)
    def _update_service_scalar_params(self, service: ServiceResourceConfig, params: Dict[str, Any]) -> None:
        """
        Update scalar service parameters.

        Args:
            service: Service configuration to update
            params: Parameters dictionary
        """
        # Update scaling factor
        if "scaling_factor" in params and isinstance(params["scaling_factor"], (int, float)):
            service.scaling_factor = float(params["scaling_factor"])

        # Update cooldown seconds
        if "cooldown_seconds" in params and isinstance(params["cooldown_seconds"], int):
            service.cooldown_seconds = params["cooldown_seconds"]

    @with_error_handling(error_class=ParameterValidationError)
    def _update_service_resources(self, service: ServiceResourceConfig, params: Dict[str, Any]) -> None:
        """
        Update service resource parameters.

        Args:
            service: Service configuration to update
            params: Parameters dictionary
        """
        # Update min resources
        if "min_resources" in params and isinstance(params["min_resources"], dict):
            self._update_resource_dict(service.min_resources, params["min_resources"], "min_resources")

        # Update max resources
        if "max_resources" in params and isinstance(params["max_resources"], dict):
            self._update_resource_dict(service.max_resources, params["max_resources"], "max_resources")

        # Update target resources
        if "target_resources" in params and isinstance(params["target_resources"], dict):
            self._update_resource_dict(service.target_resources, params["target_resources"], "target_resources")

    @with_error_handling(error_class=ParameterValidationError)
    def _update_resource_dict(
        self,
        resource_dict: Dict[ResourceType, float],
        new_values: Dict[str, Any],
        param_name: str
    ) -> None:
        """
        Update a resource dictionary with new values.

        Args:
            resource_dict: Resource dictionary to update
            new_values: New values to set
            param_name: Parameter name for error reporting
        """
        for res_type_str, res_value in new_values.items():
            try:
                res_type = ResourceType(res_type_str)
                resource_dict[res_type] = float(res_value)
            except (ValueError, TypeError) as e:
                raise ParameterValidationError(
                    message=f"Invalid resource type or value: {res_type_str}={res_value}",
                    parameter_name=f"{param_name}.{res_type_str}",
                    parameter_value=res_value
                ) from e

    def update_service_utilization(
        self, service_name: str, utilization: ResourceUtilization
    ) -> None:
        """
        Update the utilization metrics for a service.

        Args:
            service_name: Name of the service
            utilization: Resource utilization data
        """
        with self._resource_lock:
            if service_name not in self.services:
                logger.warning(f"Received utilization for unknown service: {service_name}")
                return

            if service_name not in self.utilization_history:
                self.utilization_history[service_name] = []

            # Keep history limited to avoid memory growth
            history = self.utilization_history[service_name]
            if len(history) >= 100:  # Keep last 100 data points
                history.pop(0)

            history.append(utilization)

            # Log significant changes in utilization
            if len(history) > 1:
                prev_util = history[-2]
                for res_type in ResourceType:
                    curr_value = utilization.get_utilization(res_type)
                    prev_value = prev_util.get_utilization(res_type)

                    # Log if utilization changed by more than 20%
                    if abs(curr_value - prev_value) > 20.0:
                        logger.info(
                            f"Significant {res_type.value} utilization change for {service_name}: "
                            f"{prev_value:.1f}% -> {curr_value:.1f}%"
                        )

    def set_market_regime(self, regime: str) -> None:
        """
        Set the current market regime which affects resource allocation strategy.

        Args:
            regime: Market regime (volatile, trending, ranging, normal, after_hours)
        """
        if regime not in self.market_scaling_factors:
            logger.warning(f"Unknown market regime: {regime}")
            return

        if regime != self._market_regime:
            logger.info(f"Market regime changed: {self._market_regime} -> {regime}")
            self._market_regime = regime

            # Trigger immediate resource reallocation
            if self._running:
                threading.Thread(target=self._allocate_resources, daemon=True).start()

    def get_current_allocations(self) -> Dict[str, Dict[str, float]]:
        """
        Get the current resource allocations for all services.

        Returns:
            Dictionary mapping service names to their resource allocations
        """
        result = {}
        with self._resource_lock:
            for service_name, service_config in self.services.items():
                resources = {}
                for res_type in ResourceType:
                    if res_type in service_config.target_resources:
                        resources[res_type.value] = service_config.target_resources[res_type]
                    elif res_type in service_config.min_resources:
                        resources[res_type.value] = service_config.min_resources[res_type]
                    else:
                        resources[res_type.value] = 0.0
                result[service_name] = resources
        return result

    def get_service_allocation(self, service_name: str) -> Dict[str, float]:
        """
        Get the current resource allocations for a specific service.

        Args:
            service_name: Name of the service

        Returns:
            Dictionary mapping resource types to allocation values
        """
        with self._resource_lock:
            if service_name not in self.services:
                return {}

            service_config = self.services[service_name]
            resources = {}

            for res_type in ResourceType:
                if res_type in service_config.target_resources:
                    resources[res_type.value] = service_config.target_resources[res_type]
                elif res_type in service_config.min_resources:
                    resources[res_type.value] = service_config.min_resources[res_type]
                else:
                    resources[res_type.value] = 0.0

            return resources

    def _resource_allocation_loop(self) -> None:
        """Background thread that periodically allocates resources."""
        while self._running:
            try:
                if self._metrics_provider:
                    self._update_utilization_from_metrics()
                self._allocate_resources()
            except Exception as e:
                logger.error(f"Error in resource allocation loop: {str(e)}")

            time.sleep(self.update_interval)

    def _update_utilization_from_metrics(self) -> None:
        """Update service utilization from metrics provider."""
        # This would integrate with the actual metrics provider.
        # Implementation depends on the specific metrics system used.
        # Example: Fetching from a Prometheus-like provider
        # try:
        #     for service_name in self.services.keys():
        #         # Example query: fetch CPU, memory, queue length, latency
        #         metrics = self._metrics_provider.query(
        #             f'{{service="{service_name}"}}',
        #             metrics=['cpu_usage_percent', 'memory_usage_percent', 'queue_length', 'request_latency_ms']
        #         )
        #         if metrics:
        #             utilization = ResourceUtilization(
        #                 service_name=service_name,
        #                 cpu_percent=metrics.get("cpu_usage_percent", 0.0),
        #                 memory_percent=metrics.get("memory_usage_percent", 0.0),
        #                 # disk/network might come from different queries/sources
        #                 queue_length=metrics.get("queue_length"),
        #                 request_latency_ms=metrics.get("request_latency_ms"),
        #             )
        #             self.update_service_utilization(service_name, utilization)
        # except Exception as e:
        #     logger.error(f"Error updating utilization from metrics: {str(e)}")
        # Placeholder implementation (remove in real integration)
        try:
            for service_name in self.services.keys():
                # Simulate some metrics for demonstration
                simulated_cpu = np.random.uniform(20, 70)
                simulated_mem = np.random.uniform(30, 80)
                simulated_queue = np.random.randint(0, 50) if np.random.rand() > 0.3 else None
                simulated_latency = np.random.uniform(50, 500) if np.random.rand() > 0.3 else None

                utilization = ResourceUtilization(
                    service_name=service_name,
                    cpu_percent=simulated_cpu,
                    memory_percent=simulated_mem,
                    queue_length=simulated_queue,
                    request_latency_ms=simulated_latency,
                )
                self.update_service_utilization(service_name, utilization)
        except Exception as e:
            logger.error(f"Error simulating utilization update: {str(e)}")

    def _allocate_resources(self) -> None:
        """Allocate resources based on current utilization and policies."""
        with self._resource_lock:
            # Skip if no services registered
            if not self.services:
                return

            # Calculate effective available resources (accounting for buffer)
            effective_resources = {}
            for res_type, total in self.available_resources.items():
                effective_resources[res_type] = max(0.0, total - self.resource_buffer.get(res_type, 0.0))

            # Determine allocations based on policies
            new_allocations: Dict[str, Dict[ResourceType, float]] = {}

            # Group services by policy
            policy_groups: Dict[ResourcePolicy, List[str]] = {}
            for service_name, service_config in self.services.items():
                if service_config.policy not in policy_groups:
                    policy_groups[service_config.policy] = []
                policy_groups[service_config.policy].append(service_name)

            # Process each policy group
            for policy, service_names in policy_groups.items():
                allocation_func = self.optimization_algorithms.get(policy, self._dynamic_allocation)

                # Get subset of services for this policy
                policy_services = {
                    name: self.services[name] for name in service_names
                }

                # Allocate resources for this policy group
                policy_allocations = allocation_func(policy_services, effective_resources)

                # Update the overall allocations
                new_allocations.update(policy_allocations)

            # Apply market regime scaling factors
            self._apply_market_regime_factors(new_allocations)

            # Apply load balancing if enabled
            if self.enable_load_balancing:
                self._balance_load(new_allocations, effective_resources)

            # Apply new allocations
            self._apply_allocations(new_allocations)

    def _apply_market_regime_factors(self, allocations: Dict[str, Dict[ResourceType, float]]) -> None:
        """Apply market regime scaling factors to allocations."""
        regime_factors = self.market_scaling_factors.get(self._market_regime,
                                                    self.market_scaling_factors["normal"])

        for service_name, resources in allocations.items():
            for res_type, value in list(resources.items()):
                if res_type in regime_factors:
                    # Apply regime-specific scaling factor
                    factor = regime_factors[res_type]
                    resources[res_type] = value * factor

                    # Ensure we don't exceed max resources
                    if service_name in self.services:
                        service_config = self.services[service_name]
                        if res_type in service_config.max_resources:
                            resources[res_type] = min(
                                resources[res_type],
                                service_config.max_resources[res_type]
                            )

    def _balance_load(
        self,
        allocations: Dict[str, Dict[ResourceType, float]],
        available_resources: Dict[ResourceType, float]
    ) -> None:
        """Balance load among services based on priority and utilization."""
        # This is a simplified load balancing algorithm
        # In a real system, this would be more sophisticated

        # Calculate total allocations per resource type
        total_allocated = {res_type: 0.0 for res_type in ResourceType}
        for resources in allocations.values():
            for res_type, value in resources.items():
                total_allocated[res_type] += value

        # Check for over-allocation and adjust if needed
        for res_type in ResourceType:
            if res_type in available_resources and res_type in total_allocated:
                if total_allocated[res_type] > available_resources[res_type]:
                    excess = total_allocated[res_type] - available_resources[res_type]

                    # Sort services by priority (lower priority services get reduced first)
                    priority_services = [
                        (name, self.services[name].priority.value)
                        for name in allocations.keys()
                        if name in self.services and res_type in allocations[name]
                    ]
                    priority_services.sort(key=lambda x: x[1])

                    # Reduce allocations starting from lowest priority
                    remaining_excess = excess
                    for service_name, priority in priority_services:
                        if remaining_excess <= 0:
                            break

                        service_config = self.services[service_name]
                        current_allocation = allocations[service_name][res_type]
                        min_allocation = service_config.min_resources.get(res_type, 0.0)

                        # Calculate how much can be reduced
                        reducible = max(0.0, current_allocation - min_allocation)
                        reduction = min(reducible, remaining_excess)

                        if reduction > 0:
                            allocations[service_name][res_type] -= reduction
                            remaining_excess -= reduction

                    if remaining_excess > 0:
                        logger.warning(
                            f"Could not fully balance {res_type.value} allocations. "
                            f"Still over-allocated by {remaining_excess:.2f}%"
                        )

    def _apply_allocations(self, allocations: Dict[str, Dict[ResourceType, float]]) -> None:
        """Apply new allocations to services."""
        current_time = datetime.now()

        # Track which services will be updated
        updated_services = set()

        for service_name, resources in allocations.items():
            if service_name not in self.services:
                continue

            service_config = self.services[service_name]

            # Check cooldown period
            if (service_config.last_scaling_time and
                (current_time - service_config.last_scaling_time).total_seconds() < service_config.cooldown_seconds):
                continue

            # Check if allocations actually changed
            allocation_changed = False

            for res_type, new_value in resources.items():
                current_value = service_config.target_resources.get(res_type, 0.0)

                # Only update if change is significant (>5%)
                if abs(new_value - current_value) > 5.0:
                    service_config.target_resources[res_type] = new_value
                    allocation_changed = True

            if allocation_changed:
                service_config.last_scaling_time = current_time
                updated_services.add(service_name)

                # Record the scaling decision for learning
                for res_type, new_value in resources.items():
                    old_value = service_config.target_resources.get(res_type, 0.0)
                    self.scaling_history.append({
                        'service': service_name,
                        'resource_type': res_type.value,
                        'old_value': old_value,
                        'new_value': new_value,
                        'timestamp': current_time,
                        'market_regime': self._market_regime
                    })

        # Apply updates using resource controller if available
        if self._resource_controller and updated_services:
            for service_name in updated_services:
                try:
                    service_config = self.services[service_name]
                    target_resources = service_config.target_resources
                    # Example: Interacting with a Kubernetes controller
                    # self._resource_controller.patch_deployment(
                    #     service_name,
                    #     resources={
                    #         "requests": {
                    #             "cpu": f"{target_resources.get(ResourceType.CPU, 0)}m", # Convert percentage to millicores
                    #             "memory": f"{target_resources.get(ResourceType.MEMORY, 0)}Mi" # Convert percentage to Mebibytes
                    #         },
                    #         "limits": {
                    #             "cpu": f"{service_config.max_resources.get(ResourceType.CPU, 1000)}m",
                    #             "memory": f"{service_config.max_resources.get(ResourceType.MEMORY, 1024)}Mi"
                    #         }
                    #     }
                    # )

                    # Example: Interacting with Docker API (less common for dynamic allocation)
                    # container_id = self._resource_controller.get_container_id(service_name)
                    # if container_id:
                    #     self._resource_controller.update_container(
                    #         container_id,
                    #         cpu_shares=int(target_resources.get(ResourceType.CPU, 10) * 10.24), # Example conversion
                    #         memory_limit=f"{int(target_resources.get(ResourceType.MEMORY, 100))}m" # Example conversion
                    #     )

                    # Placeholder for actual controller interaction
                    logger.info(f"Applying resource updates for {service_name}: {target_resources}")
                    # self._resource_controller.update_resources(
                    #     service_name, target_resources
                    # )
                    logger.info(f"Applied resource updates for {service_name}")
                except Exception as e:
                    logger.error(f"Failed to apply resource updates for {service_name}: {str(e)}")

    def _fixed_allocation(
        self,
        services: Dict[str, ServiceResourceConfig],
        available_resources: Dict[ResourceType, float]
    ) -> Dict[str, Dict[ResourceType, float]]:
        """
        Fixed allocation policy - allocates exact amounts specified.

        Args:
            services: Services to allocate resources for
            available_resources: Available resources

        Returns:
            Dictionary mapping service names to resource allocations
        """
        allocations = {}

        for service_name, service_config in services.items():
            allocations[service_name] = {}

            for res_type in ResourceType:
                if res_type in service_config.target_resources:
                    allocations[service_name][res_type] = service_config.target_resources[res_type]
                elif res_type in service_config.min_resources:
                    allocations[service_name][res_type] = service_config.min_resources[res_type]

        return allocations

    def _dynamic_allocation(
        self,
        services: Dict[str, ServiceResourceConfig],
        available_resources: Dict[ResourceType, float]
    ) -> Dict[str, Dict[ResourceType, float]]:
        """
        Dynamic allocation policy - scales resources based on recent utilization.

        Args:
            services: Services to allocate resources for
            available_resources: Available resources

        Returns:
            Dictionary mapping service names to resource allocations
        """
        allocations = {}

        for service_name, service_config in services.items():
            allocations[service_name] = {}

            # Get utilization history
            history = self.utilization_history.get(service_name, [])

            for res_type in ResourceType:
                # Default to minimum if specified
                if res_type in service_config.min_resources:
                    allocation = service_config.min_resources[res_type]
                else:
                    allocation = 0.0

                # Only adjust if we have utilization data
                if history:
                    # Calculate recent average utilization
                    recent_history = history[-min(len(history), 5):]  # Last 5 data points
                    avg_utilization = sum(u.get_utilization(res_type) for u in recent_history) / len(recent_history)

                    # Scale based on utilization (add 20% headroom)
                    target_allocation = avg_utilization * 1.2

                    # Apply service scaling factor
                    target_allocation *= service_config.scaling_factor

                    # Ensure we're within min/max bounds
                    if res_type in service_config.min_resources:
                        target_allocation = max(target_allocation, service_config.min_resources[res_type])
                    if res_type in service_config.max_resources:
                        target_allocation = min(target_allocation, service_config.max_resources[res_type])

                    allocation = target_allocation

                allocations[service_name][res_type] = allocation

        return allocations

    def _priority_based_allocation(
        self,
        services: Dict[str, ServiceResourceConfig],
        available_resources: Dict[ResourceType, float]
    ) -> Dict[str, Dict[ResourceType, float]]:
        """
        Priority-based allocation policy - higher priority services get resources first.

        Args:
            services: Services to allocate resources for
            available_resources: Available resources

        Returns:
            Dictionary mapping service names to resource allocations
        """
        allocations = {}

        # Initialize with minimum allocations
        for service_name, service_config in services.items():
            allocations[service_name] = {}

            for res_type in ResourceType:
                if res_type in service_config.min_resources:
                    allocations[service_name][res_type] = service_config.min_resources[res_type]
                else:
                    allocations[service_name][res_type] = 0.0

        # Calculate remaining resources after minimum allocations
        remaining_resources = available_resources.copy()
        for service_allocations in allocations.values():
            for res_type, allocation in service_allocations.items():
                if res_type in remaining_resources:
                    remaining_resources[res_type] = max(0.0, remaining_resources[res_type] - allocation)

        # Sort services by priority (highest first)
        priority_sorted = sorted(
            services.items(),
            key=lambda x: x[1].priority.value,
            reverse=True
        )

        # Distribute remaining resources by priority
        for service_name, service_config in priority_sorted:
            for res_type in ResourceType:
                if res_type not in remaining_resources:
                    continue

                if res_type not in allocations[service_name]:
                    allocations[service_name][res_type] = 0.0

                # Calculate desired additional allocation
                current = allocations[service_name][res_type]
                max_allowed = service_config.max_resources.get(res_type, float('inf'))
                desired = max_allowed - current

                # Allocate what's available, up to the desired amount
                additional = min(desired, remaining_resources[res_type])
                allocations[service_name][res_type] += additional
                remaining_resources[res_type] -= additional

        return allocations

    def _adaptive_allocation(
        self,
        services: Dict[str, ServiceResourceConfig],
        available_resources: Dict[ResourceType, float]
    ) -> Dict[str, Dict[ResourceType, float]]:
        """
        Adaptive allocation policy - uses predictive scaling and historical patterns,
        potentially incorporating queue length and latency.

        Args:
            services: Services to allocate resources for
            available_resources: Available resources

        Returns:
            Dictionary mapping service names to resource allocations
        """
        # Start with dynamic allocation as a base
        allocations = self._dynamic_allocation(services, available_resources)

        # Apply predictive scaling based on history and other metrics
        for service_name, resources in allocations.items():
            if service_name not in services:
                continue

            service_config = services[service_name]
            history = self.utilization_history.get(service_name, [])

            if len(history) < 10:  # Need enough history for prediction
                continue

            # --- Predictive Scaling ---
            # Consider CPU and Memory for predictive scaling based on utilization trend
            # TODO: Replace with a more sophisticated time series forecasting model
            # (e.g., ARIMA, Prophet, LSTM) for better accuracy.
            # This requires more historical data and potentially external libraries.
            recent_values = [u.get_utilization(ResourceType.CPU) for u in history[-10:]]

            if len(recent_values) >= 2:
                # Calculate trend (simple linear slope approximation)
                trend = (recent_values[-1] - recent_values[0]) / len(recent_values)

                # Predict future utilization based on trend
                prediction_factor = 1.0
                if trend > 1.0:  # Significant upward trend (e.g., >1% per interval)
                    prediction_factor = 1.3
                elif trend > 0.5:  # Moderate upward trend
                    prediction_factor = 1.15
                elif trend < -1.0: # Significant downward trend
                    prediction_factor = 0.9 # Allow scaling down faster
                elif trend < -0.5: # Moderate downward trend
                    prediction_factor = 0.95

                # Apply prediction factor
                resources[ResourceType.CPU] *= prediction_factor
                resources[ResourceType.MEMORY] *= prediction_factor

            # --- Queue Length / Latency Scaling ---
            # Consider scaling based on queue length or latency if available
            latest_util = history[-1]
            queue_factor = 1.0
            latency_factor = 1.0

            if latest_util.queue_length is not None and latest_util.queue_length > service_config.metadata.get("max_queue_length_threshold", 100):
                # Increase resources if queue length is high
                queue_factor = 1.2 + (latest_util.queue_length / service_config.metadata.get("max_queue_length_threshold", 100)) * 0.1 # Scale up more aggressively as queue grows
                logger.info(f"High queue length ({latest_util.queue_length}) for {service_name}, applying factor {queue_factor:.2f}")
            elif latest_util.queue_length is not None and latest_util.queue_length < service_config.metadata.get("min_queue_length_threshold", 10):
                 # Consider scaling down if queue is consistently low (less aggressive)
                 queue_factor = 0.95

            if latest_util.request_latency_ms is not None and latest_util.request_latency_ms > service_config.metadata.get("max_latency_ms_threshold", 500):
                # Increase resources if latency is high
                latency_factor = 1.15 + (latest_util.request_latency_ms / service_config.metadata.get("max_latency_ms_threshold", 500)) * 0.1
                logger.info(f"High latency ({latest_util.request_latency_ms}ms) for {service_name}, applying factor {latency_factor:.2f}")
            elif latest_util.request_latency_ms is not None and latest_util.request_latency_ms < service_config.metadata.get("min_latency_ms_threshold", 100):
                 # Consider scaling down if latency is consistently low (less aggressive)
                 latency_factor = 0.98


            # Apply queue/latency factors primarily to CPU/Memory
            combined_factor = max(queue_factor, latency_factor) # Take the more demanding factor
            if combined_factor > 1.0:
                if ResourceType.CPU in resources:
                    resources[ResourceType.CPU] *= combined_factor
                if ResourceType.MEMORY in resources:
                    resources[ResourceType.MEMORY] *= combined_factor # Memory might also be needed

            # --- Final Boundary Check ---
            for res_type in resources.keys():
                 # Ensure we're within min/max bounds after all adjustments
                if res_type in service_config.min_resources:
                    resources[res_type] = max(resources[res_type], service_config.min_resources[res_type])
                if res_type in service_config.max_resources:
                    resources[res_type] = min(resources[res_type], service_config.max_resources[res_type])

        return allocations

    def get_utilization_statistics(
        self, service_name: str, resource_type: ResourceType, window_minutes: int = 60
    ) -> Dict[str, float]:
        """
        Get statistics about resource utilization for a service.

        Args:
            service_name: Name of the service
            resource_type: Type of resource
            window_minutes: Time window in minutes

        Returns:
            Dictionary with statistics (min, max, avg, current)
        """
        with self._resource_lock:
            if service_name not in self.utilization_history:
                return {
                    "min": 0.0,
                    "max": 0.0,
                    "avg": 0.0,
                    "current": 0.0,
                    "p95": 0.0
                }

            history = self.utilization_history[service_name]
            if not history:
                return {
                    "min": 0.0,
                    "max": 0.0,
                    "avg": 0.0,
                    "current": 0.0,
                    "p95": 0.0
                }

            # Filter history by time window
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            window_history = [
                h for h in history
                if h.timestamp >= cutoff_time
            ]

            if not window_history:
                window_history = history

            values = [h.get_utilization(resource_type) for h in window_history]

            return {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "current": values[-1] if values else 0.0,
                "p95": np.percentile(values, 95) if len(values) >= 20 else max(values)
            }

    def get_allocation_efficiency(
        self, service_name: str, resource_type: ResourceType, window_minutes: int = 60
    ) -> Dict[str, float]:
        """
        Calculate resource allocation efficiency for a service.

        Args:
            service_name: Name of the service
            resource_type: Type of resource
            window_minutes: Time window in minutes

        Returns:
            Dictionary with efficiency metrics
        """
        with self._resource_lock:
            if service_name not in self.services or service_name not in self.utilization_history:
                return {
                    "utilization_ratio": 0.0,
                    "overallocation_percent": 0.0,
                    "efficiency_score": 0.0
                }

            service_config = self.services[service_name]
            current_allocation = service_config.target_resources.get(resource_type, 0.0)

            if current_allocation <= 0.0:
                return {
                    "utilization_ratio": 0.0,
                    "overallocation_percent": 0.0,
                    "efficiency_score": 0.0
                }

            stats = self.get_utilization_statistics(service_name, resource_type, window_minutes)

            utilization_ratio = stats["avg"] / current_allocation if current_allocation else 0.0
            overallocation_percent = max(0.0, 100.0 * (1.0 - utilization_ratio))

            # Calculate efficiency score (higher is better, max 100)
            # Perfect score when utilization is 80% of allocation
            if utilization_ratio <= 0.8:
                efficiency_score = utilization_ratio * 100.0 / 0.8
            else:
                # Penalize for approaching 100% utilization (risk of resource starvation)
                efficiency_score = 100.0 - ((utilization_ratio - 0.8) * 100.0)

            return {
                "utilization_ratio": utilization_ratio,
                "overallocation_percent": overallocation_percent,
                "efficiency_score": max(0.0, min(100.0, efficiency_score))
            }

    def get_overall_efficiency_report(self) -> Dict[str, Any]:
        """
        Generate overall efficiency report for all services and resources.

        Returns:
            Dictionary with efficiency metrics
        """
        report = {
            "services": {},
            "resources": {res.value: {"allocated": 0.0, "used": 0.0} for res in ResourceType},
            "overall_efficiency": 0.0
        }

        with self._resource_lock:
            # Calculate per-service metrics
            for service_name, service_config in self.services.items():
                service_report = {
                    "resources": {},
                    "average_efficiency": 0.0
                }

                efficiency_scores = []

                for res_type in ResourceType:
                    if res_type in service_config.target_resources:
                        efficiency = self.get_allocation_efficiency(service_name, res_type)
                        service_report["resources"][res_type.value] = efficiency
                        efficiency_scores.append(efficiency["efficiency_score"])

                        # Update totals
                        allocation = service_config.target_resources[res_type]
                        utilization = allocation * efficiency["utilization_ratio"]

                        report["resources"][res_type.value]["allocated"] += allocation
                        report["resources"][res_type.value]["used"] += utilization

                if efficiency_scores:
                    service_report["average_efficiency"] = sum(efficiency_scores) / len(efficiency_scores)

                report["services"][service_name] = service_report

            # Calculate overall efficiency
            efficiency_scores = []
            for res_data in report["resources"].values():
                if res_data["allocated"] > 0:
                    utilization_ratio = res_data["used"] / res_data["allocated"]

                    if utilization_ratio <= 0.8:
                        efficiency = utilization_ratio * 100.0 / 0.8
                    else:
                        efficiency = 100.0 - ((utilization_ratio - 0.8) * 100.0)

                    efficiency_scores.append(max(0.0, min(100.0, efficiency)))

                    # Add efficiency to resource data
                    res_data["efficiency"] = max(0.0, min(100.0, efficiency))

            if efficiency_scores:
                report["overall_efficiency"] = sum(efficiency_scores) / len(efficiency_scores)

        return report

    def export_config(self) -> Dict[str, Any]:
        """
        Export current configuration as a dictionary.

        Returns:
            Dictionary with current configuration
        """
        with self._resource_lock:
            config = {
                "services": [],
                "available_resources": {k.value: v for k, v in self.available_resources.items()},
                "resource_buffer": {k.value: v for k, v in self.resource_buffer.items()},
                "market_scaling_factors": self.market_scaling_factors,
                "update_interval_seconds": self.update_interval,
                "enable_predictive_scaling": self.enable_predictive_scaling,
                "enable_load_balancing": self.enable_load_balancing,
                "current_market_regime": self._market_regime
            }

            for service_name, service_config in self.services.items():
                service_data = {
                    "name": service_name,
                    "priority": service_config.priority.value,
                    "policy": service_config.policy.value,
                    "min_resources": {k.value: v for k, v in service_config.min_resources.items()},
                    "max_resources": {k.value: v for k, v in service_config.max_resources.items()},
                    "target_resources": {k.value: v for k, v in service_config.target_resources.items()},
                    "scaling_factor": service_config.scaling_factor,
                    "cooldown_seconds": service_config.cooldown_seconds,
                    "metadata": service_config.metadata.copy()
                }
                config["services"].append(service_data)

            return config

    def save_config(self, config_path: str) -> bool:
        """
        Save current configuration to a file.

        Args:
            config_path: Path to save the configuration to

        Returns:
            True if successful, False otherwise
        """
        try:
            config = self.export_config()
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {str(e)}")
            return False


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


class PrometheusMetricsProvider(MetricsProviderInterface):
    """Metrics provider that fetches data from Prometheus."""

    def __init__(self, prometheus_url: str):
        """
        Initialize the Prometheus metrics provider.

        Args:
            prometheus_url: URL of the Prometheus server
        """
        self.prometheus_url = prometheus_url
        logger.info(f"Prometheus metrics provider initialized with URL: {prometheus_url}")

    def get_metrics(self, service_name: str) -> Optional[Dict[str, float]]:
        """
        Get metrics for a service from Prometheus.

        Args:
            service_name: Name of the service

        Returns:
            Dictionary with metrics or None if unavailable
        """
        # This is a placeholder implementation
        # In a real system, this would query Prometheus API
        try:
            # Simulated metrics
            return {
                "cpu_percent": 50.0,
                "memory_percent": 60.0,
                "disk_io_percent": 30.0,
                "network_io_percent": 40.0,
                "gpu_percent": 0.0
            }
        except Exception as e:
            logger.error(f"Failed to get metrics for {service_name}: {str(e)}")
            return None


class KubernetesResourceController(ResourceControllerInterface):
    """Resource controller that updates Kubernetes resources."""

    def __init__(self, namespace: str = "default"):
        """
        Initialize the Kubernetes resource controller.

        Args:
            namespace: Kubernetes namespace
        """
        self.namespace = namespace
        logger.info(f"Kubernetes resource controller initialized for namespace: {namespace}")

    def update_resources(self, service_name: str, resources: Dict[ResourceType, float]) -> bool:
        """
        Update Kubernetes resources for a service.

        Args:
            service_name: Name of the service
            resources: Resource allocations

        Returns:
            True if successful, False otherwise
        """
        # This is a placeholder implementation
        # In a real system, this would use the Kubernetes API
        try:
            logger.info(f"Updating Kubernetes resources for {service_name}: {resources}")
            return True
        except Exception as e:
            logger.error(f"Failed to update resources for {service_name}: {str(e)}")
            return False


class DockerResourceController(ResourceControllerInterface):
    """Resource controller that updates Docker container resources."""

    def __init__(self, docker_url: str = "unix://var/run/docker.sock"):
        """
        Initialize the Docker resource controller.

        Args:
            docker_url: Docker daemon URL
        """
        self.docker_url = docker_url
        logger.info(f"Docker resource controller initialized with URL: {docker_url}")

    def update_resources(self, service_name: str, resources: Dict[ResourceType, float]) -> bool:
        """
        Update Docker resources for a service.

        Args:
            service_name: Name of the service
            resources: Resource allocations

        Returns:
            True if successful, False otherwise
        """
        # This is a placeholder implementation
        # In a real system, this would use the Docker API
        try:
            logger.info(f"Updating Docker resources for {service_name}: {resources}")
            return True
        except Exception as e:
            logger.error(f"Failed to update resources for {service_name}: {str(e)}")
            return False
