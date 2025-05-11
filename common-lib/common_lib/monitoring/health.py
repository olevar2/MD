"""
Health Module

This module provides health check functionality for the platform.
"""

import logging
import time
import asyncio
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, ClassVar, Awaitable, Union


class HealthStatus(Enum):
    """
    Health status.
    """
    
    UP = "UP"
    DOWN = "DOWN"
    DEGRADED = "DEGRADED"
    UNKNOWN = "UNKNOWN"


class HealthCheck:
    """
    Health check.
    """
    
    def __init__(
        self,
        name: str,
        check_func: Callable[[], Awaitable[HealthStatus]],
        description: Optional[str] = None,
        timeout: float = 5.0
    ):
        """
        Initialize the health check.
        
        Args:
            name: Name of the health check
            check_func: Function to check health
            description: Description of the health check
            timeout: Timeout for the health check in seconds
        """
        self.name = name
        self.check_func = check_func
        self.description = description
        self.timeout = timeout
        self.status = HealthStatus.UNKNOWN
        self.last_check_time = 0
        self.last_check_duration = 0
        self.error = None
    
    async def check(self) -> HealthStatus:
        """
        Check health.
        
        Returns:
            Health status
        """
        # Record start time
        start_time = time.time()
        
        try:
            # Execute check function with timeout
            self.status = await asyncio.wait_for(
                self.check_func(),
                timeout=self.timeout
            )
            self.error = None
        except asyncio.TimeoutError:
            # Check timed out
            self.status = HealthStatus.DOWN
            self.error = "Health check timed out"
        except Exception as e:
            # Check failed
            self.status = HealthStatus.DOWN
            self.error = str(e)
        
        # Record check time and duration
        self.last_check_time = time.time()
        self.last_check_duration = self.last_check_time - start_time
        
        return self.status
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the health check to a dictionary.
        
        Returns:
            Dictionary representation of the health check
        """
        result = {
            "name": self.name,
            "status": self.status.value,
            "last_check_time": self.last_check_time,
            "last_check_duration": self.last_check_duration
        }
        
        if self.description:
            result["description"] = self.description
        
        if self.error:
            result["error"] = self.error
        
        return result


class HealthManager:
    """
    Health manager for the platform.
    
    This class provides a singleton manager for health checks.
    """
    
    _instance: ClassVar[Optional["HealthManager"]] = None
    
    def __new__(cls, *args, **kwargs):
        """
        Create a new instance of the health manager.
        
        Returns:
            Singleton instance of the health manager
        """
        if cls._instance is None:
            cls._instance = super(HealthManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        service_name: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the health manager.
        
        Args:
            service_name: Name of the service
            logger: Logger to use (if None, creates a new logger)
        """
        # Skip initialization if already initialized
        if getattr(self, "_initialized", False):
            return
        
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.service_name = service_name
        self.health_checks = {}
        
        self._initialized = True
    
    def register_health_check(
        self,
        name: str,
        check_func: Callable[[], Awaitable[HealthStatus]],
        description: Optional[str] = None,
        timeout: float = 5.0
    ) -> HealthCheck:
        """
        Register a health check.
        
        Args:
            name: Name of the health check
            check_func: Function to check health
            description: Description of the health check
            timeout: Timeout for the health check in seconds
            
        Returns:
            Health check
        """
        health_check = HealthCheck(
            name=name,
            check_func=check_func,
            description=description,
            timeout=timeout
        )
        
        self.health_checks[name] = health_check
        
        return health_check
    
    def unregister_health_check(self, name: str) -> None:
        """
        Unregister a health check.
        
        Args:
            name: Name of the health check
        """
        if name in self.health_checks:
            del self.health_checks[name]
    
    async def check_health(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Check health.
        
        Args:
            name: Name of the health check to check (if None, checks all health checks)
            
        Returns:
            Health check result
        """
        if name:
            # Check specific health check
            if name not in self.health_checks:
                raise ValueError(f"Health check not found: {name}")
            
            health_check = self.health_checks[name]
            await health_check.check()
            
            return health_check.to_dict()
        else:
            # Check all health checks
            results = {}
            overall_status = HealthStatus.UP
            
            for health_check in self.health_checks.values():
                await health_check.check()
                results[health_check.name] = health_check.to_dict()
                
                # Update overall status
                if health_check.status == HealthStatus.DOWN:
                    overall_status = HealthStatus.DOWN
                elif health_check.status == HealthStatus.DEGRADED and overall_status != HealthStatus.DOWN:
                    overall_status = HealthStatus.DEGRADED
                elif health_check.status == HealthStatus.UNKNOWN and overall_status not in [HealthStatus.DOWN, HealthStatus.DEGRADED]:
                    overall_status = HealthStatus.UNKNOWN
            
            return {
                "status": overall_status.value,
                "service": self.service_name,
                "checks": results
            }
    
    async def is_healthy(self) -> bool:
        """
        Check if the service is healthy.
        
        Returns:
            True if the service is healthy, False otherwise
        """
        health = await self.check_health()
        return health["status"] == HealthStatus.UP.value
    
    async def is_ready(self) -> bool:
        """
        Check if the service is ready.
        
        Returns:
            True if the service is ready, False otherwise
        """
        health = await self.check_health()
        return health["status"] in [HealthStatus.UP.value, HealthStatus.DEGRADED.value]