#!/usr/bin/env python
"""
Verification script for deployed modules.

This script verifies that the standardized modules have been correctly deployed to
Data Pipeline Service and ML Integration Service.
"""

import os
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("verify_deployed_modules")

# Services to check
SERVICES = [
    {
        "name": "Data Pipeline Service",
        "path": "D:/MD/forex_trading_platform/data-pipeline-service",
        "package": "data_pipeline_service",
        "files": [
            "config/standardized_config.py",
            "logging_setup.py",
            "service_clients.py",
            "database.py",
            "error_handlers.py",
            "monitoring.py",
            "main.py",
        ],
    },
    {
        "name": "ML Integration Service",
        "path": "D:/MD/forex_trading_platform/ml-integration-service",
        "package": "ml_integration_service",
        "files": [
            "config/standardized_config.py",
            "logging_setup.py",
            "service_clients.py",
            "database.py",
            "error_handlers.py",
            "monitoring.py",
            "main.py",
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
    logger.info("Checking deployed modules")
    
    # Check files for each service
    all_files_exist = True
    for service in SERVICES:
        service_files_exist = check_files(service)
        if not service_files_exist:
            all_files_exist = False
    
    # Print overall result
    if all_files_exist:
        logger.info("All deployed modules exist")
    else:
        logger.error("Some deployed modules do not exist")


if __name__ == "__main__":
    main()