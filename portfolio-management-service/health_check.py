#!/usr/bin/env python3
"""
Health check endpoints for the service.
"""

import logging
import time
import os
import socket
import psutil
import platform
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Health check router
health_router = APIRouter(tags=["Health"])

class HealthStatus(BaseModel):
    """Health status response model."""
    status: str
    version: str
    uptime: float
    timestamp: float
    hostname: str
    details: Dict[str, Any]

class DependencyStatus(BaseModel):
    """Dependency status model."""
    name: str
    status: str
    latency: float
    details: Optional[Dict[str, Any]] = None

# Service start time
START_TIME = time.time()

# Service version
VERSION = os.environ.get("SERVICE_VERSION", "0.1.0")

@health_router.get("/health", response_model=HealthStatus)
async def health_check(request: Request) -> HealthStatus:
    """
    Basic health check endpoint.
    
    Returns:
        Health status of the service
    """
    current_time = time.time()
    uptime = current_time - START_TIME
    
    # Basic system info
    hostname = socket.gethostname()
    
    # Collect health details
    details = {
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent
        }
    }
    
    return HealthStatus(
        status="healthy",
        version=VERSION,
        uptime=uptime,
        timestamp=current_time,
        hostname=hostname,
        details=details
    )

@health_router.get("/health/liveness")
async def liveness_check() -> Dict[str, str]:
    """
    Liveness probe for Kubernetes.
    
    Returns:
        Simple status response
    """
    return {"status": "alive"}

@health_router.get("/health/readiness")
async def readiness_check() -> Dict[str, str]:
    """
    Readiness probe for Kubernetes.
    
    Returns:
        Simple status response
    """
    # TODO: Add checks for database and other dependencies
    return {"status": "ready"}
