"""
Adaptive Allocation Policy

This module implements the adaptive allocation policy, which uses predictive scaling
and historical patterns to allocate resources.
"""

import logging
import numpy as np
from typing import Dict, List

from ..models import ResourceType, ServiceResourceConfig, ResourceUtilization

# Try importing core_foundations if available, otherwise use standard logging
try:
    from core_foundations.utils.logger import get_logger
    logger = get_logger("adaptive-allocation-policy")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("adaptive-allocation-policy")


def allocate(
    services: Dict[str, ServiceResourceConfig],
    available_resources: Dict[ResourceType, float],
    utilization_history: Dict[str, List[ResourceUtilization]] = None
) -> Dict[str, Dict[ResourceType, float]]:
    """
    Adaptive allocation policy - uses predictive scaling and historical patterns.

    Args:
        services: Services to allocate resources for
        available_resources: Available resources
        utilization_history: Historical utilization data by service

    Returns:
        Dictionary mapping service names to resource allocations
    """
    from . import dynamic
    
    # Start with dynamic allocation as a base
    allocations = dynamic.allocate(services, available_resources, utilization_history)
    utilization_history = utilization_history or {}

    # Apply predictive scaling based on history and other metrics
    for service_name, resources in allocations.items():
        if service_name not in services:
            continue

        service_config = services[service_name]
        history = utilization_history.get(service_name, [])

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
            if ResourceType.CPU in resources:
                resources[ResourceType.CPU] *= prediction_factor
            if ResourceType.MEMORY in resources:
                resources[ResourceType.MEMORY] *= prediction_factor

        # --- Queue Length / Latency Scaling ---
        # Consider scaling based on queue length or latency if available
        latest_util = history[-1] if history else None
        queue_factor = 1.0
        latency_factor = 1.0

        if latest_util and latest_util.queue_length is not None:
            max_queue_threshold = service_config.metadata.get("max_queue_length_threshold", 100)
            min_queue_threshold = service_config.metadata.get("min_queue_length_threshold", 10)
            
            if latest_util.queue_length > max_queue_threshold:
                # Increase resources if queue length is high
                queue_factor = 1.2 + (latest_util.queue_length / max_queue_threshold) * 0.1 # Scale up more aggressively as queue grows
                logger.info(f"High queue length ({latest_util.queue_length}) for {service_name}, applying factor {queue_factor:.2f}")
            elif latest_util.queue_length < min_queue_threshold:
                 # Consider scaling down if queue is consistently low (less aggressive)
                 queue_factor = 0.95

        if latest_util and latest_util.request_latency_ms is not None:
            max_latency_threshold = service_config.metadata.get("max_latency_ms_threshold", 500)
            min_latency_threshold = service_config.metadata.get("min_latency_ms_threshold", 100)
            
            if latest_util.request_latency_ms > max_latency_threshold:
                # Increase resources if latency is high
                latency_factor = 1.15 + (latest_util.request_latency_ms / max_latency_threshold) * 0.1
                logger.info(f"High latency ({latest_util.request_latency_ms}ms) for {service_name}, applying factor {latency_factor:.2f}")
            elif latest_util.request_latency_ms < min_latency_threshold:
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

    logger.debug(f"Adaptive allocation policy allocated resources for {len(services)} services")
    return allocations