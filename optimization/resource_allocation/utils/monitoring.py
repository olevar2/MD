"""
Monitoring Utilities

This module provides utilities for monitoring resource allocation and utilization.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from ..models import ResourceType, ResourceUtilization

# Try importing core_foundations if available, otherwise use standard logging
try:
    from core_foundations.utils.logger import get_logger
    logger = get_logger("resource-monitoring")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("resource-monitoring")


def get_utilization_statistics(
    utilization_history: List[ResourceUtilization],
    resource_type: ResourceType,
    window_minutes: int = 60
) -> Dict[str, float]:
    """
    Get statistics about resource utilization.

    Args:
        utilization_history: List of utilization data points
        resource_type: Type of resource
        window_minutes: Time window in minutes

    Returns:
        Dictionary with statistics (min, max, avg, current, p95)
    """
    if not utilization_history:
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
        h for h in utilization_history
        if h.timestamp >= cutoff_time
    ]

    if not window_history:
        window_history = utilization_history

    values = [h.get_utilization(resource_type) for h in window_history]

    return {
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
        "current": values[-1] if values else 0.0,
        "p95": np.percentile(values, 95) if len(values) >= 20 else max(values)
    }


def get_allocation_efficiency(
    utilization_history: List[ResourceUtilization],
    resource_type: ResourceType,
    current_allocation: float,
    window_minutes: int = 60
) -> Dict[str, float]:
    """
    Calculate resource allocation efficiency.

    Args:
        utilization_history: List of utilization data points
        resource_type: Type of resource
        current_allocation: Current resource allocation
        window_minutes: Time window in minutes

    Returns:
        Dictionary with efficiency metrics
    """
    if not utilization_history or current_allocation <= 0.0:
        return {
            "utilization_ratio": 0.0,
            "overallocation_percent": 0.0,
            "efficiency_score": 0.0
        }

    stats = get_utilization_statistics(utilization_history, resource_type, window_minutes)

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


def get_overall_efficiency_report(
    services: Dict[str, Any],
    utilization_history: Dict[str, List[ResourceUtilization]]
) -> Dict[str, Any]:
    """
    Generate overall efficiency report for all services and resources.

    Args:
        services: Service configurations
        utilization_history: Utilization history by service

    Returns:
        Dictionary with efficiency metrics
    """
    report = {
        "services": {},
        "resources": {res.value: {"allocated": 0.0, "used": 0.0} for res in ResourceType},
        "overall_efficiency": 0.0
    }

    # Calculate per-service metrics
    for service_name, service_config in services.items():
        service_report = {
            "resources": {},
            "average_efficiency": 0.0
        }

        efficiency_scores = []

        for res_type in ResourceType:
            if res_type in service_config.target_resources:
                history = utilization_history.get(service_name, [])
                efficiency = get_allocation_efficiency(
                    history, 
                    res_type, 
                    service_config.target_resources[res_type]
                )
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