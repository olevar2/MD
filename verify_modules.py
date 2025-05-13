#!/usr/bin/env python
"""
Verification script for standardized modules.

This script verifies that the standardized modules exist and have the required attributes.
"""

import os
import sys
import importlib
import inspect
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("verify_modules")

# Services to verify
SERVICES = [
    {
        "name": "ML Workbench Service",
        "path": "ml-workbench-service",
        "package": "ml_workbench_service",
        "modules": [
            "config.standardized_config",
            "logging_setup",
            "service_clients",
            "database",
            "error_handlers",
            "monitoring",
            "main",
        ],
    },
    {
        "name": "Monitoring Alerting Service",
        "path": "monitoring-alerting-service",
        "package": "monitoring_alerting_service",
        "modules": [
            "config.standardized_config",
            "logging_setup",
            "service_clients",
            "database",
            "error_handlers",
            "monitoring",
            "main",
        ],
    },
]


def verify_module_existence(service: Dict[str, Any]) -> bool:
    """
    Verify that all required modules exist.

    Args:
        service: Service configuration

    Returns:
        True if all modules exist, False otherwise
    """
    logger.info(f"Verifying module existence for {service['name']}")
    
    # Add service path to sys.path
    service_path = os.path.join(os.getcwd(), service["path"])
    if service_path not in sys.path:
        sys.path.append(service_path)
    
    # Verify modules
    all_modules_exist = True
    for module_name in service["modules"]:
        full_module_name = f"{service['package']}.{module_name}"
        try:
            module = importlib.import_module(full_module_name)
            logger.info(f"Module {full_module_name} exists")
        except ImportError as e:
            logger.error(f"Module {full_module_name} does not exist: {str(e)}")
            all_modules_exist = False
    
    return all_modules_exist


def verify_config_module(service: Dict[str, Any]) -> bool:
    """
    Verify that the configuration module has the required attributes.

    Args:
        service: Service configuration

    Returns:
        True if the configuration module has the required attributes, False otherwise
    """
    logger.info(f"Verifying configuration module for {service['name']}")
    
    # Add service path to sys.path
    service_path = os.path.join(os.getcwd(), service["path"])
    if service_path not in sys.path:
        sys.path.append(service_path)
    
    # Verify configuration module
    full_module_name = f"{service['package']}.config.standardized_config"
    try:
        module = importlib.import_module(full_module_name)
        
        # Check for required attributes
        required_attributes = [
            "settings",
            "get_settings",
            "get_db_url",
            "get_api_settings",
            "get_security_settings",
            "get_monitoring_settings",
        ]
        
        all_attributes_exist = True
        for attribute in required_attributes:
            if hasattr(module, attribute):
                logger.info(f"Attribute {attribute} exists in {full_module_name}")
            else:
                logger.error(f"Attribute {attribute} does not exist in {full_module_name}")
                all_attributes_exist = False
        
        return all_attributes_exist
    except ImportError as e:
        logger.error(f"Module {full_module_name} does not exist: {str(e)}")
        return False


def verify_logging_module(service: Dict[str, Any]) -> bool:
    """
    Verify that the logging module has the required attributes.

    Args:
        service: Service configuration

    Returns:
        True if the logging module has the required attributes, False otherwise
    """
    logger.info(f"Verifying logging module for {service['name']}")
    
    # Add service path to sys.path
    service_path = os.path.join(os.getcwd(), service["path"])
    if service_path not in sys.path:
        sys.path.append(service_path)
    
    # Verify logging module
    full_module_name = f"{service['package']}.logging_setup"
    try:
        module = importlib.import_module(full_module_name)
        
        # Check for required attributes
        required_attributes = [
            "configure_logging",
            "get_logger",
            "set_correlation_id",
            "get_correlation_id",
        ]
        
        all_attributes_exist = True
        for attribute in required_attributes:
            if hasattr(module, attribute):
                logger.info(f"Attribute {attribute} exists in {full_module_name}")
            else:
                logger.error(f"Attribute {attribute} does not exist in {full_module_name}")
                all_attributes_exist = False
        
        return all_attributes_exist
    except ImportError as e:
        logger.error(f"Module {full_module_name} does not exist: {str(e)}")
        return False


def verify_monitoring_module(service: Dict[str, Any]) -> bool:
    """
    Verify that the monitoring module has the required attributes.

    Args:
        service: Service configuration

    Returns:
        True if the monitoring module has the required attributes, False otherwise
    """
    logger.info(f"Verifying monitoring module for {service['name']}")
    
    # Add service path to sys.path
    service_path = os.path.join(os.getcwd(), service["path"])
    if service_path not in sys.path:
        sys.path.append(service_path)
    
    # Verify monitoring module
    full_module_name = f"{service['package']}.monitoring"
    try:
        module = importlib.import_module(full_module_name)
        
        # Check for required attributes
        required_attributes = [
            "setup_monitoring",
            "register_health_check",
            "health_check",
            "metrics_registry",
            "start_metrics_collection",
            "stop_metrics_collection",
            "track_database_query",
            "track_service_client_request",
        ]
        
        all_attributes_exist = True
        for attribute in required_attributes:
            if hasattr(module, attribute):
                logger.info(f"Attribute {attribute} exists in {full_module_name}")
            else:
                logger.error(f"Attribute {attribute} does not exist in {full_module_name}")
                all_attributes_exist = False
        
        return all_attributes_exist
    except ImportError as e:
        logger.error(f"Module {full_module_name} does not exist: {str(e)}")
        return False


def verify_error_handlers_module(service: Dict[str, Any]) -> bool:
    """
    Verify that the error handlers module has the required attributes.

    Args:
        service: Service configuration

    Returns:
        True if the error handlers module has the required attributes, False otherwise
    """
    logger.info(f"Verifying error handlers module for {service['name']}")
    
    # Add service path to sys.path
    service_path = os.path.join(os.getcwd(), service["path"])
    if service_path not in sys.path:
        sys.path.append(service_path)
    
    # Verify error handlers module
    full_module_name = f"{service['package']}.error_handlers"
    try:
        module = importlib.import_module(full_module_name)
        
        # Check for required attributes
        required_attributes = [
            "setup_error_handlers",
            "get_error_response",
        ]
        
        all_attributes_exist = True
        for attribute in required_attributes:
            if hasattr(module, attribute):
                logger.info(f"Attribute {attribute} exists in {full_module_name}")
            else:
                logger.error(f"Attribute {attribute} does not exist in {full_module_name}")
                all_attributes_exist = False
        
        return all_attributes_exist
    except ImportError as e:
        logger.error(f"Module {full_module_name} does not exist: {str(e)}")
        return False


def verify_service_clients_module(service: Dict[str, Any]) -> bool:
    """
    Verify that the service clients module has the required attributes.

    Args:
        service: Service configuration

    Returns:
        True if the service clients module has the required attributes, False otherwise
    """
    logger.info(f"Verifying service clients module for {service['name']}")
    
    # Add service path to sys.path
    service_path = os.path.join(os.getcwd(), service["path"])
    if service_path not in sys.path:
        sys.path.append(service_path)
    
    # Verify service clients module
    full_module_name = f"{service['package']}.service_clients"
    try:
        module = importlib.import_module(full_module_name)
        
        # Check for required attributes
        required_attributes = [
            "BaseServiceClient",
            "close_all_clients",
        ]
        
        all_attributes_exist = True
        for attribute in required_attributes:
            if hasattr(module, attribute):
                logger.info(f"Attribute {attribute} exists in {full_module_name}")
            else:
                logger.error(f"Attribute {attribute} does not exist in {full_module_name}")
                all_attributes_exist = False
        
        return all_attributes_exist
    except ImportError as e:
        logger.error(f"Module {full_module_name} does not exist: {str(e)}")
        return False


def verify_database_module(service: Dict[str, Any]) -> bool:
    """
    Verify that the database module has the required attributes.

    Args:
        service: Service configuration

    Returns:
        True if the database module has the required attributes, False otherwise
    """
    logger.info(f"Verifying database module for {service['name']}")
    
    # Add service path to sys.path
    service_path = os.path.join(os.getcwd(), service["path"])
    if service_path not in sys.path:
        sys.path.append(service_path)
    
    # Verify database module
    full_module_name = f"{service['package']}.database"
    try:
        module = importlib.import_module(full_module_name)
        
        # Check for required attributes
        required_attributes = [
            "Base",
            "init_db",
            "create_tables",
            "get_sync_session",
            "get_async_session",
            "BaseRepository",
        ]
        
        all_attributes_exist = True
        for attribute in required_attributes:
            if hasattr(module, attribute):
                logger.info(f"Attribute {attribute} exists in {full_module_name}")
            else:
                logger.error(f"Attribute {attribute} does not exist in {full_module_name}")
                all_attributes_exist = False
        
        return all_attributes_exist
    except ImportError as e:
        logger.error(f"Module {full_module_name} does not exist: {str(e)}")
        return False


def verify_api_endpoints(service: Dict[str, Any]) -> bool:
    """
    Verify that the API endpoints exist.

    Args:
        service: Service configuration

    Returns:
        True if the API endpoints exist, False otherwise
    """
    logger.info(f"Verifying API endpoints for {service['name']}")
    
    # Add service path to sys.path
    service_path = os.path.join(os.getcwd(), service["path"])
    if service_path not in sys.path:
        sys.path.append(service_path)
    
    # Verify API endpoints
    api_endpoints = []
    if service["name"] == "ML Workbench Service":
        api_endpoints = [
            "api.v1.model_registry",
            "api.v1.model_training",
            "api.v1.model_serving",
            "api.v1.model_monitoring",
            "api.v1.transfer_learning",
        ]
    elif service["name"] == "Monitoring Alerting Service":
        api_endpoints = [
            "api.v1.alerts",
            "api.v1.dashboards",
            "api.v1.prometheus",
            "api.v1.alertmanager",
            "api.v1.grafana",
            "api.v1.notifications",
        ]
    
    all_endpoints_exist = True
    for endpoint in api_endpoints:
        full_module_name = f"{service['package']}.{endpoint}"
        try:
            module = importlib.import_module(full_module_name)
            logger.info(f"API endpoint {full_module_name} exists")
        except ImportError as e:
            logger.error(f"API endpoint {full_module_name} does not exist: {str(e)}")
            all_endpoints_exist = False
    
    return all_endpoints_exist


def verify_service(service: Dict[str, Any]) -> bool:
    """
    Verify that a service has all the required modules and attributes.

    Args:
        service: Service configuration

    Returns:
        True if the service has all the required modules and attributes, False otherwise
    """
    logger.info(f"Verifying service {service['name']}")
    
    # Verify modules
    modules_exist = verify_module_existence(service)
    
    # Verify configuration module
    config_module_valid = verify_config_module(service)
    
    # Verify logging module
    logging_module_valid = verify_logging_module(service)
    
    # Verify monitoring module
    monitoring_module_valid = verify_monitoring_module(service)
    
    # Verify error handlers module
    error_handlers_module_valid = verify_error_handlers_module(service)
    
    # Verify service clients module
    service_clients_module_valid = verify_service_clients_module(service)
    
    # Verify database module
    database_module_valid = verify_database_module(service)
    
    # Verify API endpoints
    api_endpoints_valid = verify_api_endpoints(service)
    
    # Return overall result
    return (
        modules_exist
        and config_module_valid
        and logging_module_valid
        and monitoring_module_valid
        and error_handlers_module_valid
        and service_clients_module_valid
        and database_module_valid
        and api_endpoints_valid
    )


def main():
    """Main function."""
    logger.info("Verifying standardized modules")
    
    # Verify services
    all_services_valid = True
    for service in SERVICES:
        service_valid = verify_service(service)
        if not service_valid:
            all_services_valid = False
    
    # Print overall result
    if all_services_valid:
        logger.info("All services have valid standardized modules")
    else:
        logger.error("Some services have invalid standardized modules")


if __name__ == "__main__":
    main()