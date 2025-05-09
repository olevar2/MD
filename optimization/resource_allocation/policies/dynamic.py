"""
Dynamic Allocation Policy

This module implements the dynamic allocation policy, which scales resources based
on recent utilization.
"""

import logging
from typing import Dict, List

from ..models import ResourceType, ServiceResourceConfig, ResourceUtilization

# Try importing core_foundations if available, otherwise use standard logging
try:
    from core_foundations.utils.logger import get_logger
    logger = get_logger("dynamic-allocation-policy")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("dynamic-allocation-policy")


def allocate(
    services: Dict[str, ServiceResourceConfig],
    available_resources: Dict[ResourceType, float],
    utilization_history: Dict[str, List[ResourceUtilization]] = None
) -> Dict[str, Dict[ResourceType, float]]:
    """
    Dynamic allocation policy - scales resources based on recent utilization.

    Args:
        services: Services to allocate resources for
        available_resources: Available resources
        utilization_history: Historical utilization data by service

    Returns:
        Dictionary mapping service names to resource allocations
    """
    allocations = {}
    utilization_history = utilization_history or {}

    for service_name, service_config in services.items():
        allocations[service_name] = {}

        # Get utilization history
        history = utilization_history.get(service_name, [])

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

    logger.debug(f"Dynamic allocation policy allocated resources for {len(services)} services")
    return allocations