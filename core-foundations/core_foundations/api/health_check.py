"""
Health Check API Functionality.

Provides standardized health check endpoints for all services with
comprehensive dependency checks, resource monitoring, and metrics reporting.
"""

import datetime
import platform
import socket
import psutil
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from fastapi import APIRouter, Depends, FastAPI, Response, status

from core_foundations.models.schemas import HealthCheckResponse, HealthStatus, ResourceMetrics, DependencyStatus
from core_foundations.utils.logger import get_logger

logger = get_logger("health_check")


class HealthCheck:
    """
    Health check functionality for services.
    
    Provides standardized health check endpoints and functions with comprehensive
    monitoring of service health, dependencies, and system resources.
    """
    
    def __init__(self, service_name: str, version: str):
        """
        Initialize the health check.
        
        Args:
            service_name: Name of the service
            version: Version of the service
        """
        self.service_name = service_name
        self.version = version
        self.start_time = time.time()
        self.checks: List[Dict[str, Any]] = []
        self.dependencies: Dict[str, Callable[[], Tuple[HealthStatus, float, Optional[Dict[str, Any]]]]] = {}
        
        # Add default system checks
        self.add_check("memory_usage", self._check_memory_usage, critical=False)
        self.add_check("disk_usage", self._check_disk_usage, critical=True)
    
    def add_check(self, name: str, check_func: Callable[[], bool], critical: bool = True):
        """
        Add a health check function.
        
        Args:
            name: Name of the check
            check_func: Function that returns True if check passes, False otherwise
            critical: Whether this check is critical for service health
        """
        self.checks.append({
            "name": name,
            "check_func": check_func,
            "critical": critical
        })
        logger.info(f"Added health check: {name}")
    
    def add_dependency(self, name: str, check_func: Callable[[], Tuple[HealthStatus, float, Optional[Dict[str, Any]]]]):
        """
        Add a dependency health check.
        
        Args:
            name: Name of the dependency
            check_func: Function that returns a tuple of (HealthStatus, response_time_ms, details)
        """
        self.dependencies[name] = check_func
        logger.info(f"Added dependency check: {name}")
    
    def _check_memory_usage(self) -> bool:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            # Consider service healthy if memory usage is below 90%
            return memory.percent < 90.0
        except Exception as e:
            logger.warning(f"Failed to check memory usage: {str(e)}")
            return False
    
    def _check_disk_usage(self) -> bool:
        """Check disk space usage."""
        try:
            disk = psutil.disk_usage('/')
            # Consider service healthy if disk usage is below 85%
            return disk.percent < 85.0
        except Exception as e:
            logger.warning(f"Failed to check disk usage: {str(e)}")
            return False
            
    def get_resource_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return ResourceMetrics(
                cpu_usage=psutil.cpu_percent(),
                memory_usage=memory.percent,
                memory_available=memory.available,
                disk_usage=disk.percent,
                disk_available=disk.free,
                uptime=int(time.time() - psutil.boot_time())
            )
        except Exception as e:
            logger.error(f"Failed to get resource metrics: {str(e)}")
            # Return default values if metrics collection fails
            return ResourceMetrics(
                cpu_usage=0.0,
                memory_usage=0.0,
                memory_available=0,
                disk_usage=0.0,
                disk_available=0,
                uptime=0
            )
    
    def check_health(self) -> HealthCheckResponse:
        """
        Perform all health checks and return comprehensive health information.
        
        Returns:
            HealthCheckResponse with detailed health status information
        """
        status = HealthStatus.HEALTHY
        system_info = {
            "hostname": socket.gethostname(),
            "system_info": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "processors": os.cpu_count()
            }
        }
        
        checks_result = {}
        dependencies_result = {}
        
        # Calculate service uptime
        uptime = int(time.time() - self.start_time)
        
        # Get resource metrics
        try:
            resources = self.get_resource_metrics()
        except Exception as e:
            logger.error(f"Failed to get resource metrics: {str(e)}")
            resources = None
            
        # Run health checks
        for check in self.checks:
            try:
                result = check["check_func"]()
                checks_result[check["name"]] = result
                
                if not result and check["critical"]:
                    status = HealthStatus.UNHEALTHY
                    logger.warning(f"Critical health check failed: {check['name']}")
                elif not result:
                    status = HealthStatus.DEGRADED if status == HealthStatus.HEALTHY else status
                    logger.warning(f"Non-critical health check failed: {check['name']}")
                    
            except Exception as e:
                logger.error(f"Health check failed: {check['name']}", exc_info=e)
                checks_result[check["name"]] = {
                    "status": "error",
                    "error": str(e)
                }
                
        # Check dependencies
        for name, check_func in self.dependencies.items():
            try:
                start_time = time.time()
                dep_status, response_time, details = check_func()
                # If response time wasn't provided, calculate it
                if response_time <= 0:
                    response_time = (time.time() - start_time) * 1000  # Convert to ms
                
                timestamp = datetime.datetime.utcnow().isoformat() + "Z"
                
                dependencies_result[name] = DependencyStatus(
                    status=dep_status,
                    response_time=response_time,
                    details=details,
                    last_checked=timestamp
                ).dict()
                
                # Update overall status based on dependency status
                if dep_status == HealthStatus.UNHEALTHY and status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                    logger.warning(f"Dependency unhealthy: {name}")
                
            except Exception as e:
                logger.error(f"Dependency check failed: {name}", exc_info=e)
                timestamp = datetime.datetime.utcnow().isoformat() + "Z"
                
                dependencies_result[name] = DependencyStatus(
                    status=HealthStatus.UNHEALTHY,
                    response_time=0.0,
                    details={"error": str(e)},
                    last_checked=timestamp
                ).dict()
        
        # Create and return the final response
        return HealthCheckResponse(
            status=status,
            service=self.service_name,
            version=self.version,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            uptime=uptime,
            resources=resources,
            dependencies=dependencies_result,
            details=system_info,
            checks=checks_result
        )


def create_health_router(health_check: HealthCheck) -> APIRouter:
    """
    Create a FastAPI router with health check endpoints.
    
    Args:
        health_check: HealthCheck instance
        
    Returns:
        APIRouter with health check endpoints
    """
    router = APIRouter(tags=["Health"])
    
    @router.get("/health", response_model=HealthCheckResponse)
    async def health():
        """Get comprehensive service health status."""
        return health_check.check_health()
    
    @router.get("/health/live", status_code=200)
    async def liveness():
        """
        Liveness probe - simple check that service is running.
        
        Returns 200 OK if service is alive, otherwise 503 Service Unavailable.
        """
        return {"status": "alive", "timestamp": datetime.datetime.utcnow().isoformat() + "Z"}
    
    @router.get("/health/ready")
    async def readiness(response: Response):
        """
        Readiness probe - check if service is ready to accept requests.
        
        Returns 200 OK if service is ready, otherwise 503 Service Unavailable.
        """
        health_status = health_check.check_health()
        if health_status.status == HealthStatus.UNHEALTHY:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "not ready", "reason": "Service is unhealthy"}
        return {"status": "ready"}
    
    return router


def add_health_check_to_app(
    app: FastAPI,
    service_name: str,
    version: str,
    checks: Optional[List[Dict[str, Any]]] = None,
    dependencies: Optional[Dict[str, Callable[[], Tuple[HealthStatus, float, Optional[Dict[str, Any]]]]]] = None
) -> HealthCheck:
    """
    Add health check endpoints to a FastAPI application.
    
    Args:
        app: FastAPI application
        service_name: Name of the service
        version: Version of the service
        checks: List of checks to add
        dependencies: Dictionary of dependency checks to add
        
    Returns:
        HealthCheck instance
    """
    health_check = HealthCheck(service_name, version)
    
    # Add checks
    if checks:
        for check in checks:
            health_check.add_check(
                check["name"],
                check["check_func"],
                check.get("critical", True)
            )
    
    # Add dependencies
    if dependencies:
        for name, check_func in dependencies.items():
            health_check.add_dependency(name, check_func)
    
    # Create and include router
    health_router = create_health_router(health_check)
    app.include_router(health_router, prefix="/api/v1")
    
    return health_check