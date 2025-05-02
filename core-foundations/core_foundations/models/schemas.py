"""
Core Foundation Schemas Module.

Contains Pydantic schemas used throughout the core foundations module.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthStatus(str, Enum):
    """Health status enum for health check responses."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ResourceMetrics(BaseModel):
    """System resource metrics model for health checks."""
    
    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    memory_available: int = Field(..., description="Available memory in bytes")
    disk_usage: float = Field(..., description="Disk usage percentage")
    disk_available: int = Field(..., description="Available disk space in bytes")
    uptime: int = Field(..., description="System uptime in seconds")


class DependencyStatus(BaseModel):
    """Status of a service dependency."""
    
    status: HealthStatus
    response_time: float = Field(..., description="Response time in milliseconds")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details about the dependency")
    last_checked: str = Field(..., description="ISO formatted timestamp of last check")


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoints."""
    
    status: HealthStatus
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: str = Field(..., description="ISO formatted timestamp")
    uptime: int = Field(..., description="Service uptime in seconds")
    resources: ResourceMetrics = Field(None, description="Resource metrics")
    dependencies: Dict[str, DependencyStatus] = Field(
        default_factory=dict, description="Status of dependencies"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional health check details"
    )
    checks: Dict[str, Any] = Field(
        default_factory=dict, description="Results of individual health checks"
    )
