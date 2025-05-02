"""
resource_allocator.py

Purpose: Implement dynamic resource allocation strategies for services based on priority and current load.

This module provides the ResourceAllocator class, which is responsible for managing
and distributing computational resources (CPU, memory) among various services
within the forex trading platform. It employs priority-based allocation, adaptive
scaling based on real-time metrics, and fair sharing policies to optimize
performance and prevent resource starvation.

Integration Points:
- Monitoring Services (e.g., Prometheus, Datadog): Fetches real-time system and service metrics.
- Container Management Platforms (e.g., Kubernetes API, Docker API): Adjusts resource limits/requests
  and scales service replicas.
- Service Orchestration Components: Used by controllers or custom logic to manage resources.
- Auto-scaling Components: Provides data and logic for informed scaling decisions.
"""

import time
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from prometheus_client import Counter, Gauge, Summary
from kubernetes import client as k8s_client
from kubernetes.client import ApiException
from kubernetes.client.rest import ApiException as K8sApiException
from kubernetes.config import load_kube_config, load_incluster_config
import requests

from core_foundations.utils.logger import get_logger

class PrometheusMetricsClient:
    """
    Client for collecting metrics from Prometheus.
    
    Attributes:
        base_url: The base URL for the Prometheus API
        logger: Logger instance
    """
    
    def __init__(self, prometheus_url: str = "http://prometheus-server:9090"):
        """
        Initialize the Prometheus metrics client.
        
        Args:
            prometheus_url: The base URL for the Prometheus API
        """
        self.base_url = prometheus_url
        self.logger = get_logger("prometheus_metrics_client")
        self.session = requests.Session()
        
        # Define standard metrics we'll use
        self.metrics_mapping = {
            "cpu_utilization": 'sum(rate(container_cpu_usage_seconds_total{pod=~"%s.*"}[5m])) / sum(container_spec_cpu_quota{pod=~"%s.*"} / 100000)',
            "memory_usage_gb": 'sum(container_memory_usage_bytes{pod=~"%s.*"}) / 1024^3',
            "request_queue_length": 'sum(request_queue_length{service="%s"})',
            "error_rate": 'sum(rate(http_requests_total{service="%s",status=~"5.."}[5m])) / sum(rate(http_requests_total{service="%s"}[5m]))'
        }
        
        # Test connection
        try:
            self.query("up")
            self.logger.info(f"Successfully connected to Prometheus at {prometheus_url}")
        except Exception as e:
            self.logger.warning(f"Could not connect to Prometheus at {prometheus_url}: {e}")

    def query(self, query_str: str) -> Dict[str, Any]:
        """
        Execute a PromQL query.
        
        Args:
            query_str: The PromQL query to execute
            
        Returns:
            The query result as a dictionary
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/query",
                params={"query": query_str},
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            if result["status"] != "success":
                self.logger.error(f"Prometheus query failed: {result.get('error', 'Unknown error')}")
                return {}
                
            return result
        except requests.RequestException as e:
            self.logger.error(f"Error querying Prometheus: {e}")
            return {}
    
    def get_service_metrics(self, service: str) -> Dict[str, float]:
        """
        Get metrics for a specific service.
        
        Args:
            service: The name of the service
            
        Returns:
            Dictionary of metrics with their values
        """
        metrics = {}
        
        for metric_name, query_template in self.metrics_mapping.items():
            try:
                if "%s" in query_template:
                    # If query contains multiple placeholders for service name
                    if query_template.count("%s") > 1:
                        query_str = query_template % (service, service)
                    else:
                        query_str = query_template % service
                else:
                    query_str = query_template
                    
                result = self.query(query_str)
                
                if result and "data" in result and "result" in result["data"] and result["data"]["result"]:
                    # Extract the value (assuming single result)
                    value = float(result["data"]["result"][0]["value"][1])
                    metrics[metric_name] = value
            except Exception as e:
                self.logger.error(f"Error getting {metric_name} for {service}: {e}")
        
        return metrics


class KubernetesClient:
    """
    Client for interacting with Kubernetes APIs.
    
    Attributes:
        logger: Logger instance
        v1: Kubernetes CoreV1Api client
        apps_v1: Kubernetes AppsV1Api client
    """
    
    def __init__(self, in_cluster: bool = True, kubeconfig_path: Optional[str] = None):
        """
        Initialize the Kubernetes client.
        
        Args:
            in_cluster: Whether the client is running inside a Kubernetes cluster
            kubeconfig_path: Path to the kubeconfig file (if not using in-cluster config)
        """
        self.logger = get_logger("kubernetes_client")
        
        try:
            if in_cluster:
                load_incluster_config()
                self.logger.info("Using in-cluster Kubernetes configuration")
            else:
                load_kube_config(kubeconfig_path)
                self.logger.info(f"Using kubeconfig from {kubeconfig_path or 'default location'}")
                
            # Initialize API clients
            self.v1 = k8s_client.CoreV1Api()
            self.apps_v1 = k8s_client.AppsV1Api()
            
            # Test connection
            self.v1.list_namespace(limit=1)
            self.logger.info("Successfully connected to Kubernetes API")
        except Exception as e:
            self.logger.error(f"Failed to initialize Kubernetes client: {e}")
            # Still create the client objects even if we can't connect
            self.v1 = k8s_client.CoreV1Api() 
            self.apps_v1 = k8s_client.AppsV1Api()
    
    def get_service_replicas(self, service: str, namespace: str = "default") -> int:
        """
        Get current replica count for a service.
        
        Args:
            service: The name of the service (deployment)
            namespace: The Kubernetes namespace
            
        Returns:
            Current replica count, or 1 if the deployment is not found
        """
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=service,
                namespace=namespace
            )
            return deployment.spec.replicas
        except K8sApiException as e:
            if e.status == 404:
                self.logger.warning(f"Deployment {service} not found in namespace {namespace}")
            else:
                self.logger.error(f"Error getting replica count for {service}: {e}")
            return 1
    
    def scale_service(self, service: str, replicas: int, namespace: str = "default") -> bool:
        """
        Scale a service to the specified number of replicas.
        
        Args:
            service: The name of the service (deployment)
            replicas: The desired number of replicas
            namespace: The Kubernetes namespace
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Patch the deployment with the new replica count
            patch = {"spec": {"replicas": replicas}}
            self.apps_v1.patch_namespaced_deployment(
                name=service,
                namespace=namespace,
                body=patch
            )
            self.logger.info(f"Scaled {service} to {replicas} replicas")
            return True
        except Exception as e:
            self.logger.error(f"Error scaling {service}: {e}")
            return False
    
    def update_service_resources(self, service: str, resources: Dict[str, str], namespace: str = "default") -> bool:
        """
        Update resource limits/requests for a service.
        
        Args:
            service: The name of the service (deployment)
            resources: Dictionary with resource settings (e.g., {'cpu_request': '1.5', 'memory_limit': '4Gi'})
            namespace: The Kubernetes namespace
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First, get the current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=service,
                namespace=namespace
            )
            
            # Map from our API to Kubernetes resource format
            resource_mapping = {
                'cpu_request': ('requests', 'cpu'),
                'memory_request': ('requests', 'memory'),
                'cpu_limit': ('limits', 'cpu'),
                'memory_limit': ('limits', 'memory')
            }
            
            # Apply updates
            for resource_name, value in resources.items():
                if resource_name in resource_mapping:
                    resource_type, k8s_resource = resource_mapping[resource_name]
                    
                    # Make sure containers exist
                    if not deployment.spec.template.spec.containers:
                        self.logger.error(f"No containers found in deployment {service}")
                        return False
                    
                    # Update each container
                    for container in deployment.spec.template.spec.containers:
                        # Initialize resources if needed
                        if not container.resources:
                            container.resources = k8s_client.V1ResourceRequirements(
                                requests={}, limits={}
                            )
                        
                        # Initialize the specific resource dict if needed
                        resource_dict = getattr(container.resources, resource_type) or {}
                        
                        # Update the resource
                        resource_dict[k8s_resource] = value
                        setattr(container.resources, resource_type, resource_dict)
            
            # Update the deployment
            self.apps_v1.patch_namespaced_deployment(
                name=service,
                namespace=namespace,
                body=deployment
            )
            
            self.logger.info(f"Updated resources for {service}: {resources}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating resources for {service}: {e}")
            return False
            
    def get_cluster_capacity(self) -> Dict[str, float]:
        """
        Get the total capacity of the Kubernetes cluster.
        
        Returns:
            Dictionary with cluster capacity metrics
        """
        try:
            nodes = self.v1.list_node().items
            cpu_capacity = 0.0
            memory_gb = 0.0
            
            for node in nodes:
                # Skip nodes that are not ready
                if not any(condition.type == "Ready" and condition.status == "True" 
                          for condition in node.status.conditions):
                    continue
                    
                cpu = node.status.capacity.get('cpu')
                memory = node.status.capacity.get('memory')
                
                # Convert memory from Ki to GB
                if cpu:
                    cpu_capacity += float(cpu)
                if memory and memory.endswith('Ki'):
                    memory_gb += float(memory[:-2]) / (1024 * 1024)
            
            return {"cpu": cpu_capacity, "memory_gb": memory_gb}
        except Exception as e:
            self.logger.error(f"Error getting cluster capacity: {e}")
            return {"cpu": 0.0, "memory_gb": 0.0}


# Define Prometheus metrics for ResourceAllocator
RESOURCE_ALLOCATION_SUMMARY = Summary(
    'resource_allocator_allocation_duration_seconds', 
    'Time taken for resource allocation cycle',
    ['outcome']
)

RESOURCE_ADJUSTMENT_COUNT = Counter(
    'resource_allocator_adjustments_total', 
    'Number of resource adjustments made',
    ['service', 'resource_type', 'adjustment_type'] # adjustment_type: scale_up, scale_down, resource_update
)

RESOURCE_ALLOCATION_FAILURES = Counter(
    'resource_allocator_failures_total', 
    'Number of resource allocation failures',
    ['service', 'reason'] # reason: metrics_fetch_error, scaling_error, resource_update_error
)

SERVICE_RESOURCE_ALLOCATION = Gauge(
    'service_resource_allocation', 
    'Current resource allocation for a service',
    ['service', 'resource_type'] # resource_type: cpu_request, cpu_limit, memory_request, memory_limit, replicas
)


class ResourceAllocator:
    """
    Manages dynamic resource allocation for platform services.
    """

    def __init__(self, 
                monitoring_client: Optional[PrometheusMetricsClient] = None, 
                container_client: Optional[KubernetesClient] = None,
                service_priorities: Dict[str, int] = None,
                namespace: str = "default"):
        """
        Initializes the ResourceAllocator.

        Args:
            monitoring_client: Client instance for fetching metrics from the monitoring system.
            container_client: Client instance for interacting with the container management platform.
            service_priorities: A dictionary mapping service names to their priority levels (e.g., higher number = higher priority).
            namespace: Kubernetes namespace where services are running.
        """
        self.monitoring_client = monitoring_client or PrometheusMetricsClient()
        self.container_client = container_client or KubernetesClient()
        self.service_priorities = service_priorities or {}
        self.namespace = namespace
        self.current_allocations: Dict[str, Dict[str, Any]] = {} # Stores current resource settings per service
        self.logger = get_logger("resource_allocator")
        self.logger.info("ResourceAllocator initialized.")
        
    def get_load_metrics(self, services: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Fetches real-time load metrics for specified services from the monitoring system.

        Args:
            services: A list of service names to fetch metrics for.

        Returns:
            A dictionary mapping service names to their load metrics (e.g.,
            {'cpu_utilization': 0.75, 'memory_usage_gb': 2.5, 'request_queue_length': 150}).
            Returns an empty dict for services where metrics are unavailable.
        """
        self.logger.info(f"Fetching load metrics for services: {services}")
        all_metrics = {}
        for service in services:
            try:
                metrics = self.monitoring_client.get_service_metrics(service)
                if metrics:
                    all_metrics[service] = metrics
                else:
                    self.logger.warning(f"No metrics found for service: {service}")
                    RESOURCE_ALLOCATION_FAILURES.labels(service=service, reason='metrics_fetch_error').inc()
            except Exception as e:
                self.logger.error(f"Error fetching metrics for {service}: {e}")
                RESOURCE_ALLOCATION_FAILURES.labels(service=service, reason='metrics_fetch_error').inc()
                all_metrics[service] = {} # Return empty dict for this service on error

        return all_metrics

    def calculate_resource_needs(self, service_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
        """
        Calculates the required resources (CPU, memory, replicas) for each service
        based on their current load metrics and priority.

        Args:
            service_metrics: A dictionary mapping service names to their load metrics.

        Returns:
            A dictionary mapping service names to their calculated resource needs
            (e.g., {'cpu_request': '1.2', 'memory_limit': '3Gi', 'replicas': 5}).
        """
        self.logger.info("Calculating resource needs based on metrics.")
        resource_needs = {}

        for service, metrics in service_metrics.items():
            # Placeholder: Implement sophisticated calculation logic here
            # This example uses simple thresholds and scaling factors
            needs = {}
            current_replicas = self.container_client.get_service_replicas(service, self.namespace)

            # Replica scaling based on CPU utilization or queue length
            target_replicas = current_replicas
            cpu_util = metrics.get("cpu_utilization", 0.0)
            queue_len = metrics.get("request_queue_length", 0.0)

            if cpu_util > 0.8: # Scale up if CPU > 80%
                target_replicas = min(current_replicas + 2, 10) # Example: Add 2 replicas, max 10
            elif queue_len > 100 * current_replicas: # Scale up if queue length is high
                 target_replicas = min(current_replicas + 1, 10) # Example: Add 1 replica, max 10
            elif cpu_util < 0.3 and queue_len < 10 * current_replicas and current_replicas > 1: # Scale down if CPU < 30% and queue low
                target_replicas = max(current_replicas - 1, 1) # Example: Remove 1 replica, min 1

            if target_replicas != current_replicas:
                needs['replicas'] = target_replicas

            # Resource request/limit adjustments (example)
            # needs['cpu_request'] = f"{cpu_util + 0.2:.1f}" # Example: Request slightly more than current usage
            # needs['memory_limit'] = f"{metrics.get('memory_usage_gb', 1.0) * 1.5:.1f}Gi" # Example: Limit 1.5x current usage

            if needs:
                resource_needs[service] = needs

        self.logger.debug(f"Calculated resource needs: {resource_needs}")
        return resource_needs

    def apply_allocations(self, resource_needs: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
        """
        Applies the calculated resource allocations to the services via the container
        management platform.

        Args:
            resource_needs: A dictionary mapping service names to their desired resource settings.

        Returns:
            A dictionary mapping service names to a boolean indicating if the allocation
            was successfully applied.
        """
        self.logger.info(f"Applying resource allocations: {resource_needs}")
        results = {}
        start_time = time.time()

        for service, needs in resource_needs.items():
            success = True
            resource_updates = {}
            scale_update = None

            # Separate scaling from resource updates
            if 'replicas' in needs:
                scale_update = needs['replicas']

            for key, value in needs.items():
                if key != 'replicas':
                    resource_updates[key] = str(value) # Ensure values are strings for K8s

            # Apply resource updates first
            if resource_updates:
                if self.container_client.update_service_resources(service, resource_updates, self.namespace):
                    RESOURCE_ADJUSTMENT_COUNT.labels(service=service, resource_type='limits_requests', adjustment_type='resource_update').inc()
                    # Update gauge metrics
                    for res_type, val_str in resource_updates.items():
                         try:
                             # Attempt to convert to float for gauge, handle units like 'Gi', 'm'
                             val_float = float(val_str.rstrip('GgiMm')) # Basic unit stripping
                             SERVICE_RESOURCE_ALLOCATION.labels(service=service, resource_type=res_type).set(val_float)
                         except ValueError:
                             self.logger.warning(f"Could not convert resource value {val_str} to float for gauge.")
                else:
                    success = False
                    RESOURCE_ALLOCATION_FAILURES.labels(service=service, reason='resource_update_error').inc()

            # Apply scaling updates
            if scale_update is not None and success: # Only scale if resource update (if any) succeeded
                current_replicas = self.container_client.get_service_replicas(service, self.namespace)
                if scale_update != current_replicas:
                    if self.container_client.scale_service(service, scale_update, self.namespace):
                        adjustment_type = 'scale_up' if scale_update > current_replicas else 'scale_down'
                        RESOURCE_ADJUSTMENT_COUNT.labels(service=service, resource_type='replicas', adjustment_type=adjustment_type).inc()
                        SERVICE_RESOURCE_ALLOCATION.labels(service=service, resource_type='replicas').set(scale_update)
                    else:
                        success = False
                        RESOURCE_ALLOCATION_FAILURES.labels(service=service, reason='scaling_error').inc()
                else:
                     # Ensure gauge reflects current state even if no change needed
                     SERVICE_RESOURCE_ALLOCATION.labels(service=service, resource_type='replicas').set(current_replicas)


            results[service] = success
            if success:
                 self.current_allocations[service] = needs # Update internal state on success

        duration = time.time() - start_time
        outcome = 'success' if all(results.values()) else 'partial_failure'
        RESOURCE_ALLOCATION_SUMMARY.labels(outcome=outcome).observe(duration)
        self.logger.info(f"Finished applying allocations in {duration:.2f}s. Results: {results}")
        return results

    def run_allocation_cycle(self, services: List[str]):
        """
        Executes a full resource allocation cycle: fetch metrics, calculate needs, apply allocations.

        Args:
            services: List of service names to manage.
        """
        self.logger.info(f"Starting resource allocation cycle for services: {services}")
        try:
            metrics = self.get_load_metrics(services)
            needs = self.calculate_resource_needs(metrics)
            self.apply_allocations(needs)
            self.logger.info("Resource allocation cycle completed.")
        except Exception as e:
            self.logger.error(f"Resource allocation cycle failed: {e}", exc_info=True)
            RESOURCE_ALLOCATION_SUMMARY.labels(outcome='failure').observe(time.time() - start_time) # Assuming start_time is accessible or re-get

