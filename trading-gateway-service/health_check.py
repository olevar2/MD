#!/usr/bin/env python3
"""
Health Check Module

This module provides health check functionality for the service.
"""

import logging
import sys
import os
import json
import time
import socket
import datetime
import platform
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_system_info() -> Dict[str, Any]:
    """
    Get system information.
    
    Returns:
        Dictionary with system information
    """
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "time": datetime.datetime.now().isoformat()
    }


def get_service_status() -> Dict[str, Any]:
    """
    Get service status.
    
    Returns:
        Dictionary with service status
    """
    return {
        "status": "UP",
        "service": "trading-gateway-service",
        "version": "1.0.0",
        "uptime": time.time()
    }


def health_check() -> Dict[str, Any]:
    """
    Perform health check.
    
    Returns:
        Health check result
    """
    system_info = get_system_info()
    service_status = get_service_status()
    
    return {
        "system": system_info,
        "service": service_status,
        "status": "healthy"
    }


if __name__ == "__main__":
    """Run health check when script is executed directly."""
    result = health_check()
    print(json.dumps(result, indent=2))
    sys.exit(0)
