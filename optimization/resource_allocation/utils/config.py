"""
Configuration Utilities

This module provides utilities for loading and validating resource allocation
configuration.
"""

import json
import logging
import os
import traceback
from typing import Dict, List, Any, Optional

from ..models import ServiceResourceConfig, ServicePriority, ResourcePolicy, ResourceType

# Try importing core_foundations if available, otherwise use standard logging
try:
    from core_foundations.utils.logger import get_logger
    logger = get_logger("resource-config")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("resource-config")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load resource allocation configuration from a JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary with configuration data
    """
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config_data
    except Exception as e:
        error_details = {
            "config_path": config_path,
            "traceback": traceback.format_exc()
        }
        logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
        raise ValueError(f"Failed to load configuration: {str(e)}") from e


def process_service_configs(service_configs: List[Dict[str, Any]]) -> Dict[str, ServiceResourceConfig]:
    """
    Process service configurations from loaded config data.

    Args:
        service_configs: List of service configuration dictionaries

    Returns:
        Dictionary mapping service names to ServiceResourceConfig objects
    """
    services = {}
    
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

        services[name] = service_resource_config
    
    return services


def load_resource_settings(config_data: Dict[str, Any]) -> Dict[str, Dict[ResourceType, float]]:
    """
    Load resource settings from config data.

    Args:
        config_data: Configuration data dictionary

    Returns:
        Dictionary with resource settings
    """
    settings = {
        "available_resources": {},
        "resource_buffer": {}
    }
    
    # Load available resources if specified
    if "available_resources" in config_data:
        for resource_type_str, value in config_data["available_resources"].items():
            try:
                resource_type = ResourceType(resource_type_str)
                settings["available_resources"][resource_type] = float(value)
            except (ValueError, TypeError):
                logger.warning(f"Invalid resource type or value: {resource_type_str}={value}")
                continue

    # Load resource buffer if specified
    if "resource_buffer" in config_data:
        for resource_type_str, value in config_data["resource_buffer"].items():
            try:
                resource_type = ResourceType(resource_type_str)
                settings["resource_buffer"][resource_type] = float(value)
            except (ValueError, TypeError):
                logger.warning(f"Invalid resource type or value: {resource_type_str}={value}")
                continue
    
    return settings


def export_config(
    services: Dict[str, ServiceResourceConfig],
    available_resources: Dict[ResourceType, float],
    resource_buffer: Dict[ResourceType, float],
    market_scaling_factors: Dict[str, Dict[ResourceType, float]],
    update_interval_seconds: int,
    enable_predictive_scaling: bool,
    enable_load_balancing: bool,
    current_market_regime: str
) -> Dict[str, Any]:
    """
    Export current configuration as a dictionary.

    Args:
        services: Service configurations
        available_resources: Available resources
        resource_buffer: Resource buffer
        market_scaling_factors: Market regime scaling factors
        update_interval_seconds: Update interval
        enable_predictive_scaling: Whether predictive scaling is enabled
        enable_load_balancing: Whether load balancing is enabled
        current_market_regime: Current market regime

    Returns:
        Dictionary with current configuration
    """
    config = {
        "services": [],
        "available_resources": {k.value: v for k, v in available_resources.items()},
        "resource_buffer": {k.value: v for k, v in resource_buffer.items()},
        "market_scaling_factors": market_scaling_factors,
        "update_interval_seconds": update_interval_seconds,
        "enable_predictive_scaling": enable_predictive_scaling,
        "enable_load_balancing": enable_load_balancing,
        "current_market_regime": current_market_regime
    }

    for service_name, service_config in services.items():
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


def save_config(config_data: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to a file.

    Args:
        config_data: Configuration data
        config_path: Path to save the configuration to

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {str(e)}")
        return False