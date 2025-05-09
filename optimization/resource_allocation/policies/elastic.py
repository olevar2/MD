"""
Elastic Allocation Policy

This module implements the elastic allocation policy, which provides quick response
to resource demands with rapid scaling.
"""

import logging
from typing import Dict, List

from ..models import ResourceType, ServiceResourceConfig, ResourceUtilization

# Try importing core_foundations if available, otherwise use standard logging
try:
    from core_foundations.utils.logger import get_logger
    logger = get_logger("elastic-allocation-policy")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("elastic-allocation-policy")


def allocate(
    services: Dict[str, ServiceResourceConfig],
    available_resources: Dict[ResourceType, float],
    utilization_history: Dict[str, List[ResourceUtilization]] = None
) -> Dict[str, Dict[ResourceType, float]]:
    """
    Elastic allocation policy - quick response to resource demands.

    Args:
        services: Services to allocate resources for
        available_resources: Available resources
        utilization_history: Historical utilization data by service

    Returns:
        Dictionary mapping service names to resource allocations
    """
    from . import adaptive
    
    # Start with adaptive allocation as a base
    allocations = adaptive.allocate(services, available_resources, utilization_history)
    utilization_history = utilization_history or {}

    # Apply more aggressive scaling factors for elastic policy
    for service_name, resources in allocations.items():
        if service_name not in services:
            continue

        service_config = services[service_name]
        history = utilization_history.get(service_name, [])

        # Skip if no history
        if not history:
            continue

        # Get the most recent utilization
        latest_util = history[-1]
        
        # Get the previous allocation if available (from target_resources)
        previous_allocation = {}
        for res_type in ResourceType:
            if res_type in service_config.target_resources:
                previous_allocation[res_type] = service_config.target_resources[res_type]
        
        # If we have previous allocations, compare with current utilization
        if previous_allocation:
            for res_type in ResourceType:
                if res_type not in resources or res_type not in previous_allocation:
                    continue
                
                current_util = latest_util.get_utilization(res_type)
                prev_alloc = previous_allocation[res_type]
                
                # Calculate utilization percentage of allocation
                util_percentage = current_util / prev_alloc if prev_alloc > 0 else 0
                
                # Apply elastic scaling rules
                if util_percentage > 0.9:  # High utilization (>90% of allocation)
                    # Scale up aggressively (50% increase)
                    scale_factor = 1.5
                    resources[res_type] = min(
                        resources[res_type] * scale_factor,
                        service_config.max_resources.get(res_type, float('inf'))
                    )
                    logger.info(f"Elastic scaling up {service_name} {res_type.value} by factor {scale_factor}")
                elif util_percentage < 0.3:  # Low utilization (<30% of allocation)
                    # Scale down moderately (20% decrease)
                    scale_factor = 0.8
                    resources[res_type] = max(
                        resources[res_type] * scale_factor,
                        service_config.min_resources.get(res_type, 0.0)
                    )
                    logger.info(f"Elastic scaling down {service_name} {res_type.value} by factor {scale_factor}")
        
        # Apply burst handling for queue spikes
        if latest_util.queue_length is not None:
            burst_threshold = service_config.metadata.get("burst_queue_threshold", 50)
            if latest_util.queue_length > burst_threshold:
                # Apply immediate burst capacity (double CPU allocation)
                if ResourceType.CPU in resources:
                    burst_factor = 2.0
                    resources[ResourceType.CPU] = min(
                        resources[ResourceType.CPU] * burst_factor,
                        service_config.max_resources.get(ResourceType.CPU, float('inf'))
                    )
                    logger.info(f"Applying burst capacity to {service_name} due to queue spike: {latest_util.queue_length}")

    logger.debug(f"Elastic allocation policy allocated resources for {len(services)} services")
    return allocations