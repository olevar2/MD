"""
Resource Allocator Module

This module implements a ResourceAllocator class for dynamically managing and allocating system resources
(CPU, memory) based on priority, load, and predefined policies. Includes adaptive scaling logic and
resource usage optimization algorithms.

The ResourceAllocator is designed to be used by service orchestration components (like Kubernetes controllers
or custom schedulers) to efficiently distribute resources across the trading platform components.

Note: This is a facade module that re-exports the refactored implementation from the resource_allocation package.
"""

# Re-export from the refactored package
from optimization.resource_allocation import (
    ResourceType,
    ServicePriority,
    ResourcePolicy,
    ServiceResourceConfig,
    ResourceUtilization,
    ScalingDecision,
    ResourceAllocationResult,
    ResourceAllocator,
    ResourceControllerInterface,
    KubernetesResourceController,
    DockerResourceController,
    MetricsProviderInterface,
    PrometheusMetricsProvider
)

# For backward compatibility
from optimization.resource_allocation.controllers.interface import ResourceControllerInterface as ResourceControllerInterface
from optimization.resource_allocation.metrics.interface import MetricsProviderInterface as MetricsProviderInterface
from optimization.resource_allocation.controllers.kubernetes import KubernetesResourceController as KubernetesResourceController
from optimization.resource_allocation.controllers.docker import DockerResourceController as DockerResourceController

__all__ = [
    'ResourceType',
    'ServicePriority',
    'ResourcePolicy',
    'ServiceResourceConfig',
    'ResourceUtilization',
    'ScalingDecision',
    'ResourceAllocationResult',
    'ResourceAllocator',
    'ResourceControllerInterface',
    'KubernetesResourceController',
    'DockerResourceController',
    'MetricsProviderInterface',
    'PrometheusMetricsProvider'
]