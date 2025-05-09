"""
Fixed Allocation Policy

This module implements the fixed allocation policy, which allocates exact amounts
specified in the service configuration.
"""

import logging
from typing import Dict

from ..models import ResourceType, ServiceResourceConfig

# Try importing core_foundations if available, otherwise use standard logging
try:
    from core_foundations.utils.logger import get_logger
    logger = get_logger("fixed-allocation-policy")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("fixed-allocation-policy")


def allocate(
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

    logger.debug(f"Fixed allocation policy allocated resources for {len(services)} services")
    return allocations