"""
Priority-Based Allocation Policy

This module implements the priority-based allocation policy, which allocates resources
to higher priority services first.
"""

import logging
from typing import Dict

from ..models import ResourceType, ServiceResourceConfig

# Try importing core_foundations if available, otherwise use standard logging
try:
    from core_foundations.utils.logger import get_logger
    logger = get_logger("priority-allocation-policy")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("priority-allocation-policy")


def allocate(
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

    logger.debug(f"Priority-based allocation policy allocated resources for {len(services)} services")
    return allocations