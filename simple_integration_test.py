#!/usr/bin/env python
"""
Simple Integration Test Script for Forex Trading Platform.

This script tests the health endpoints of all services to verify they are working correctly.
"""

import os
import sys
import time
import logging
import requests
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("simple_integration_test")

# Service configurations
SERVICES = [
    {
        "name": "ML Workbench Service",
        "url": "http://localhost:8030",
        "health_endpoint": "/health",
    },
    {
        "name": "Monitoring Alerting Service",
        "url": "http://localhost:8009",
        "health_endpoint": "/health",
    },
    {
        "name": "Data Pipeline Service",
        "url": "http://localhost:8010",
        "health_endpoint": "/health",
    },
    {
        "name": "ML Integration Service",
        "url": "http://localhost:8020",
        "health_endpoint": "/health",
    },
]


def check_health(service: Dict[str, Any]) -> bool:
    """
    Check the health of a service.

    Args:
        service: Service configuration

    Returns:
        True if the service is healthy, False otherwise
    """
    logger.info(f"Checking health of {service['name']}...")
    
    try:
        response = requests.get(f"{service['url']}{service['health_endpoint']}")
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok":
                logger.info(f"{service['name']} is healthy")
                return True
        
        logger.error(f"{service['name']} is not healthy: {response.text}")
        return False
    except requests.exceptions.ConnectionError:
        logger.error(f"{service['name']} is not reachable")
        return False


def check_root(service: Dict[str, Any]) -> bool:
    """
    Check the root endpoint of a service.

    Args:
        service: Service configuration

    Returns:
        True if the service is healthy, False otherwise
    """
    logger.info(f"Checking root endpoint of {service['name']}...")
    
    try:
        response = requests.get(f"{service['url']}/")
        if response.status_code == 200:
            data = response.json()
            if data.get("service") == service["name"]:
                logger.info(f"{service['name']} root endpoint is working")
                return True
        
        logger.error(f"{service['name']} root endpoint is not working: {response.text}")
        return False
    except requests.exceptions.ConnectionError:
        logger.error(f"{service['name']} is not reachable")
        return False


def run_tests() -> bool:
    """
    Run all tests.

    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("Running integration tests...")
    
    all_passed = True
    for service in SERVICES:
        if not check_health(service):
            all_passed = False
        
        if not check_root(service):
            all_passed = False
    
    if all_passed:
        logger.info("All tests passed")
    else:
        logger.error("Some tests failed")
    
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)