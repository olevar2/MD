"""
Resource Allocation Package

This package provides tools for dynamically managing and allocating system resources
(CPU, memory) based on priority, load, and predefined policies.

Public API:
    ResourceAllocator: Main class for resource allocation
    ResourceType: Enum representing different resource types
    ServicePriority: Enum representing service priority levels
    ResourcePolicy: Enum representing resource allocation policies
    ServiceResourceConfig: Data model for service resource configuration
    ResourceUtilization: Data model for resource utilization
    ScalingDecision: Data model for scaling decisions
    ResourceAllocationResult: Data model for allocation results
    ResourceControllerInterface: Interface for resource controllers
    KubernetesResourceController: Kubernetes resource controller
    DockerResourceController: Docker resource controller
    MetricsProviderInterface: Interface for metrics providers
    PrometheusMetricsProvider: Prometheus metrics provider
"""

from optimization.resource_allocation.models import (
    ResourceType,
    ServicePriority,
    ResourcePolicy,
    ServiceResourceConfig,
    ResourceUtilization,
    ScalingDecision,
    ResourceAllocationResult
)
from optimization.resource_allocation.allocator import ResourceAllocator
from optimization.resource_allocation.controllers.interface import ResourceControllerInterface
from optimization.resource_allocation.controllers.kubernetes import KubernetesResourceController
from optimization.resource_allocation.controllers.docker import DockerResourceController
from optimization.resource_allocation.metrics.interface import MetricsProviderInterface
from optimization.resource_allocation.metrics.prometheus import PrometheusMetricsProvider

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