#!/usr/bin/env python
"""
Check if the required files exist.

This script checks if the required files for the standardized modules exist.
"""

import os
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("check_files")

# Services to check
SERVICES = [
    {
        "name": "ML Workbench Service",
        "path": "D:/MD/forex_trading_platform/ml_workbench-service",
        "package": "ml_workbench_service",
        "files": [
            "config/standardized_config.py",
            "logging_setup.py",
            "service_clients.py",
            "database.py",
            "error_handlers.py",
            "monitoring.py",
            "main.py",
            "api/v1/model_registry.py",
            "api/v1/model_training.py",
            "api/v1/model_serving.py",
            "api/v1/model_monitoring.py",
            "api/v1/transfer_learning.py",
        ],
    },
    {
        "name": "Monitoring Alerting Service",
        "path": "D:/MD/forex_trading_platform/monitoring-alerting-service",
        "package": "monitoring_alerting_service",
        "files": [
            "config/standardized_config.py",
            "logging_setup.py",
            "service_clients.py",
            "database.py",
            "error_handlers.py",
            "monitoring.py",
            "main.py",
            "api/v1/alerts.py",
            "api/v1/dashboards.py",
            "api/v1/prometheus.py",
            "api/v1/alertmanager.py",
            "api/v1/grafana.py",
            "api/v1/notifications.py",
        ],
    },
]


def check_files(service: Dict[str, Any]) -> bool:
    """
    Check if the required files for a service exist.

    Args:
        service: Service configuration

    Returns:
        True if all files exist, False otherwise
    """
    logger.info(f"Checking files for {service['name']}")
    
    # Check files
    all_files_exist = True
    for file_path in service["files"]:
        full_path = os.path.join(service["path"], service["package"], file_path)
        if os.path.isfile(full_path):
            logger.info(f"File {full_path} exists")
        else:
            logger.error(f"File {full_path} does not exist")
            all_files_exist = False
    
    return all_files_exist


def main():
    """Main function."""
    logger.info("Checking files")
    
    # Check files for each service
    all_files_exist = True
    for service in SERVICES:
        service_files_exist = check_files(service)
        if not service_files_exist:
            all_files_exist = False
    
    # Print overall result
    if all_files_exist:
        logger.info("All files exist")
    else:
        logger.error("Some files do not exist")


if __name__ == "__main__":
    main()