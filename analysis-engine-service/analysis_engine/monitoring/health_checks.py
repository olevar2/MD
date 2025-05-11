"""
Enhanced Health Checks Module for Analysis Engine Service.

This module provides comprehensive health check capabilities for the service
and its dependencies, including detailed component health checks and
dependency health checks.
"""

import asyncio
import logging
import time
import psutil
import os
from typing import Dict, Any, List, Callable, Awaitable, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel

from analysis_engine.monitoring.metrics import MetricsRecorder, DEPENDENCY_HEALTH

logger = logging.getLogger(__name__)

class HealthStatus(str, Enum):
    """Health status enum."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class ComponentHealth(BaseModel):
    """Component health model."""
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = {}
    last_check_time: datetime = datetime.utcnow()

class DependencyHealth(BaseModel):
    """Dependency health model."""
    name: str
    status: HealthStatus
    latency_ms: float = 0
    message: str = ""
    details: Dict[str, Any] = {}
    last_check_time: datetime = datetime.utcnow()

class ServiceHealth(BaseModel):
    """Service health model."""
    status: HealthStatus
    components: List[ComponentHealth] = []
    dependencies: List[DependencyHealth] = []
    uptime_seconds: float
    version: str
    start_time: datetime
    current_time: datetime = datetime.utcnow()
    resource_usage: Dict[str, float] = {}

class HealthCheck:
    """Health check manager for the service."""

    def __init__(self, service_name: str, version: str):
        """
        Initialize the health check manager.

        Args:
            service_name: Name of the service
            version: Service version
        """
        self.service_name = service_name
        self.version = version
        self.start_time = datetime.utcnow()
        self.component_checks: Dict[str, Callable[[], Awaitable[ComponentHealth]]] = {}
        self.dependency_checks: Dict[str, Callable[[], Awaitable[DependencyHealth]]] = {}

    def add_component_check(self, name: str, check_func: Callable[[], Awaitable[ComponentHealth]]) -> None:
        """
        Add a component health check.

        Args:
            name: Component name
            check_func: Async function that returns a ComponentHealth object
        """
        self.component_checks[name] = check_func

    def add_dependency_check(self, name: str, check_func: Callable[[], Awaitable[DependencyHealth]]) -> None:
        """
        Add a dependency health check.

        Args:
            name: Dependency name
            check_func: Async function that returns a DependencyHealth object
        """
        self.dependency_checks[name] = check_func

    async def check_health(self) -> ServiceHealth:
        """
        Check the health of the service and its components and dependencies.

        Returns:
            ServiceHealth object
        """
        # Check component health
        component_results = await asyncio.gather(
            *[check_func() for check_func in self.component_checks.values()],
            return_exceptions=True
        )

        components = []
        for i, result in enumerate(component_results):
            if isinstance(result, Exception):
                component_name = list(self.component_checks.keys())[i]
                logger.error(f"Error checking component {component_name}: {result}")
                components.append(ComponentHealth(
                    name=component_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Error during health check: {str(result)}"
                ))
            else:
                components.append(result)

        # Check dependency health
        dependency_results = await asyncio.gather(
            *[check_func() for check_func in self.dependency_checks.values()],
            return_exceptions=True
        )

        dependencies = []
        for i, result in enumerate(dependency_results):
            if isinstance(result, Exception):
                dependency_name = list(self.dependency_checks.keys())[i]
                logger.error(f"Error checking dependency {dependency_name}: {result}")
                dependencies.append(DependencyHealth(
                    name=dependency_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Error during health check: {str(result)}"
                ))
            else:
                dependencies.append(result)
                # Update dependency health metric
                MetricsRecorder.record_dependency_health(
                    result.name,
                    result.status == HealthStatus.HEALTHY
                )

        # Calculate uptime
        uptime = (datetime.utcnow() - self.start_time).total_seconds()

        # Get resource usage
        resource_usage = self._get_resource_usage()

        # Determine overall status
        status = self._determine_overall_status(components, dependencies)

        return ServiceHealth(
            status=status,
            components=components,
            dependencies=dependencies,
            uptime_seconds=uptime,
            version=self.version,
            start_time=self.start_time,
            resource_usage=resource_usage
        )

    def _determine_overall_status(
        self,
        components: List[ComponentHealth],
        dependencies: List[DependencyHealth]
    ) -> HealthStatus:
        """
        Determine the overall health status based on component and dependency health.

        Args:
            components: List of component health results
            dependencies: List of dependency health results

        Returns:
            Overall health status
        """
        # Check for unhealthy components or dependencies
        if any(c.status == HealthStatus.UNHEALTHY for c in components):
            return HealthStatus.UNHEALTHY

        if any(d.status == HealthStatus.UNHEALTHY for d in dependencies):
            return HealthStatus.UNHEALTHY

        # Check for degraded components or dependencies
        if any(c.status == HealthStatus.DEGRADED for c in components):
            return HealthStatus.DEGRADED

        if any(d.status == HealthStatus.DEGRADED for d in dependencies):
            return HealthStatus.DEGRADED

        # All checks passed
        return HealthStatus.HEALTHY

    def _get_resource_usage(self) -> Dict[str, float]:
        """
        Get resource usage information.

        Returns:
            Dictionary of resource usage metrics
        """
        process = psutil.Process(os.getpid())

        # Get CPU usage
        cpu_percent = process.cpu_percent(interval=0.1)

        # Get memory usage
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()

        # Get disk usage
        disk_usage = psutil.disk_usage('/')

        # Update resource usage metrics
        MetricsRecorder.record_resource_usage('cpu', cpu_percent)
        MetricsRecorder.record_resource_usage('memory', memory_percent)
        MetricsRecorder.record_resource_usage('disk', disk_usage.percent)

        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_rss_mb': memory_info.rss / (1024 * 1024),
            'memory_vms_mb': memory_info.vms / (1024 * 1024),
            'disk_percent': disk_usage.percent
        }

# Utility functions for creating health checks

async def check_database_connection(
    db_client: Any = None,
    name: str = "database"
) -> DependencyHealth:
    """
    Check database connection health.

    Args:
        db_client: Database client (optional, if None will use the standard connection module)
        name: Dependency name

    Returns:
        DependencyHealth object
    """
    start_time = time.time()

    try:
        if db_client is not None:
            # Execute a simple query to check connection using provided client
            await db_client.execute("SELECT 1")
        else:
            # Use the standard connection module
            from analysis_engine.db.connection import check_async_db_connection

            # Check connection
            connection_ok = await check_async_db_connection()

            if not connection_ok:
                raise Exception("Database connection check failed")

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        return DependencyHealth(
            name=name,
            status=HealthStatus.HEALTHY,
            latency_ms=latency_ms,
            message="Database connection is healthy",
            details={
                "latency_ms": latency_ms
            }
        )
    except Exception as e:
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        logger.error(f"Database health check failed: {e}")

        return DependencyHealth(
            name=name,
            status=HealthStatus.UNHEALTHY,
            latency_ms=latency_ms,
            message=f"Database connection failed: {str(e)}",
            details={
                "error": str(e),
                "latency_ms": latency_ms
            }
        )

async def check_redis_connection(
    redis_client: Any,
    name: str = "redis"
) -> DependencyHealth:
    """
    Check Redis connection health.

    Args:
        redis_client: Redis client
        name: Dependency name

    Returns:
        DependencyHealth object
    """
    start_time = time.time()

    try:
        # Execute a simple command to check connection
        await redis_client.ping()

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        return DependencyHealth(
            name=name,
            status=HealthStatus.HEALTHY,
            latency_ms=latency_ms,
            message="Redis connection is healthy",
            details={
                "latency_ms": latency_ms
            }
        )
    except Exception as e:
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        logger.error(f"Redis health check failed: {e}")

        return DependencyHealth(
            name=name,
            status=HealthStatus.UNHEALTHY,
            latency_ms=latency_ms,
            message=f"Redis connection failed: {str(e)}",
            details={
                "error": str(e),
                "latency_ms": latency_ms
            }
        )

async def check_service_connection(
    service_url: str,
    service_name: str,
    timeout: float = 5.0,
    http_client = None
) -> DependencyHealth:
    """
    Check external service connection health.

    Args:
        service_url: URL of the service
        service_name: Name of the service
        timeout: Request timeout in seconds
        http_client: HTTP client to use (optional)

    Returns:
        DependencyHealth object
    """
    import aiohttp

    start_time = time.time()

    try:
        # Create a client if not provided
        if http_client is None:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{service_url}/health/live",
                    timeout=timeout
                ) as response:
                    await response.text()

                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000

                    if response.status == 200:
                        return DependencyHealth(
                            name=service_name,
                            status=HealthStatus.HEALTHY,
                            latency_ms=latency_ms,
                            message=f"{service_name} connection is healthy",
                            details={
                                "latency_ms": latency_ms,
                                "status_code": response.status
                            }
                        )
                    else:
                        return DependencyHealth(
                            name=service_name,
                            status=HealthStatus.DEGRADED,
                            latency_ms=latency_ms,
                            message=f"{service_name} returned non-200 status: {response.status}",
                            details={
                                "latency_ms": latency_ms,
                                "status_code": response.status
                            }
                        )
        else:
            # Use provided client
            async with http_client.get(
                f"{service_url}/health/live",
                timeout=timeout
            ) as response:
                await response.text()

                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000

                if response.status == 200:
                    return DependencyHealth(
                        name=service_name,
                        status=HealthStatus.HEALTHY,
                        latency_ms=latency_ms,
                        message=f"{service_name} connection is healthy",
                        details={
                            "latency_ms": latency_ms,
                            "status_code": response.status
                        }
                    )
                else:
                    return DependencyHealth(
                        name=service_name,
                        status=HealthStatus.DEGRADED,
                        latency_ms=latency_ms,
                        message=f"{service_name} returned non-200 status: {response.status}",
                        details={
                            "latency_ms": latency_ms,
                            "status_code": response.status
                        }
                    )
    except asyncio.TimeoutError:
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        logger.error(f"{service_name} health check timed out after {timeout} seconds")

        return DependencyHealth(
            name=service_name,
            status=HealthStatus.UNHEALTHY,
            latency_ms=latency_ms,
            message=f"{service_name} connection timed out after {timeout} seconds",
            details={
                "error": "Timeout",
                "latency_ms": latency_ms,
                "timeout": timeout
            }
        )
    except Exception as e:
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        logger.error(f"{service_name} health check failed: {e}")

        return DependencyHealth(
            name=service_name,
            status=HealthStatus.UNHEALTHY,
            latency_ms=latency_ms,
            message=f"{service_name} connection failed: {str(e)}",
            details={
                "error": str(e),
                "latency_ms": latency_ms
            }
        )

async def check_component_health(
    component: Any,
    name: str
) -> ComponentHealth:
    """
    Check component health.

    Args:
        component: Component to check
        name: Component name

    Returns:
        ComponentHealth object
    """
    try:
        # Check if component has a health_check method
        if hasattr(component, 'health_check') and callable(component.health_check):
            start_time = time.time()

            # Call the health_check method
            if asyncio.iscoroutinefunction(component.health_check):
                result = await component.health_check()
            else:
                result = component.health_check()

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            # Check the result
            if isinstance(result, bool):
                # Simple boolean result
                if result:
                    return ComponentHealth(
                        name=name,
                        status=HealthStatus.HEALTHY,
                        message=f"{name} is healthy",
                        details={
                            "latency_ms": latency_ms
                        }
                    )
                else:
                    return ComponentHealth(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"{name} is unhealthy",
                        details={
                            "latency_ms": latency_ms
                        }
                    )
            elif isinstance(result, dict):
                # Dictionary result
                status = result.get('status', 'unknown')
                message = result.get('message', '')
                details = result.get('details', {})

                if isinstance(details, dict):
                    details['latency_ms'] = latency_ms
                else:
                    details = {'latency_ms': latency_ms, 'original_details': details}

                if status == 'healthy':
                    return ComponentHealth(
                        name=name,
                        status=HealthStatus.HEALTHY,
                        message=message or f"{name} is healthy",
                        details=details
                    )
                elif status == 'degraded':
                    return ComponentHealth(
                        name=name,
                        status=HealthStatus.DEGRADED,
                        message=message or f"{name} is degraded",
                        details=details
                    )
                else:
                    return ComponentHealth(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=message or f"{name} is unhealthy",
                        details=details
                    )
            else:
                # Unknown result format
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.DEGRADED,
                    message=f"{name} health check returned unknown format",
                    details={
                        "latency_ms": latency_ms,
                        "result": str(result)
                    }
                )
        else:
            # No health_check method
            return ComponentHealth(
                name=name,
                status=HealthStatus.DEGRADED,
                message=f"{name} does not implement health_check method",
                details={}
            )
    except Exception as e:
        logger.error(f"Error checking component {name}: {e}")

        return ComponentHealth(
            name=name,
            status=HealthStatus.UNHEALTHY,
            message=f"Error checking {name}: {str(e)}",
            details={
                "error": str(e)
            }
        )
