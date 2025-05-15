"""
Service Registry

This module provides a registry for service discovery.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, List, Set

from common_lib.config.config_manager import ConfigManager


class ServiceRegistry:
    """
    Registry for service discovery.
    """

    def __init__(self):
        """
        Initialize the service registry.
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config_manager = ConfigManager()

        # Get configuration
        try:
            service_specific = self.config_manager.get_service_specific_config()
            if hasattr(service_specific, "services"):
                self.services_config = getattr(service_specific, "services")
            else:
                self.services_config = {}
        except Exception as e:
            self.logger.warning(f"Error getting services configuration: {str(e)}")
            self.services_config = {}

        # Initialize services
        self.services = {}
        for service_name, service_config in self.services_config.items():
            self.services[service_name] = {
                "name": service_name,
                "url": service_config.get("url", ""),
                "health_check_url": service_config.get("health_check_url", ""),
                "health_check_interval": service_config.get("health_check_interval", 60),
                "timeout": service_config.get("timeout", 30),
                "retry": service_config.get("retry", {
                    "retries": 3,
                    "delay": 1.0,
                    "backoff": 2.0
                }),
                "circuit_breaker": service_config.get("circuit_breaker", {
                    "failure_threshold": 5,
                    "recovery_timeout": 30
                }),
                "status": "unknown",
                "last_check": 0,
                "endpoints": service_config.get("endpoints", [])
            }

        # Initialize health check task
        self.health_check_task = None

    async def start(self):
        """
        Start the service registry.
        """
        # Start health check task
        self.health_check_task = asyncio.create_task(self._health_check_loop())

    async def stop(self):
        """
        Stop the service registry.
        """
        # Stop health check task
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

    async def _health_check_loop(self):
        """
        Health check loop.
        """
        while True:
            try:
                # Check all services
                for service_name, service in self.services.items():
                    # Check if it's time to check the service
                    now = time.time()
                    if now - service["last_check"] >= service["health_check_interval"]:
                        # Check service health
                        await self._check_service_health(service_name)
            except Exception as e:
                self.logger.error(f"Error in health check loop: {str(e)}")

            # Sleep for a while
            await asyncio.sleep(1)

    async def _check_service_health(self, service_name: str):
        """
        Check the health of a service.

        Args:
            service_name: Service name
        """
        # Get service
        service = self.services.get(service_name)
        if not service:
            return

        # Update last check time
        service["last_check"] = time.time()

        # Check if health check URL is configured
        health_check_url = service["health_check_url"]
        if not health_check_url:
            # No health check URL, assume service is healthy
            service["status"] = "healthy"
            return

        # Check service health
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(health_check_url)
                if response.status_code == 200:
                    service["status"] = "healthy"
                else:
                    service["status"] = "unhealthy"
        except Exception as e:
            self.logger.error(f"Error checking service health: {str(e)}")
            service["status"] = "unhealthy"

    def get_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a service by name.

        Args:
            service_name: Service name

        Returns:
            Service information, or None if the service is not found
        """
        return self.services.get(service_name)

    def get_service_url(self, service_name: str) -> str:
        """
        Get the URL for a service.

        Args:
            service_name: Service name

        Returns:
            Service URL

        Raises:
            ValueError: If the service is not found
        """
        service = self.get_service(service_name)
        if not service:
            raise ValueError(f"Service {service_name} not found")

        return service["url"]

    def get_service_status(self, service_name: str) -> str:
        """
        Get the status of a service.

        Args:
            service_name: Service name

        Returns:
            Service status

        Raises:
            ValueError: If the service is not found
        """
        service = self.get_service(service_name)
        if not service:
            raise ValueError(f"Service {service_name} not found")

        return service["status"]

    def get_all_services(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all services.

        Returns:
            All services
        """
        return self.services

    def get_healthy_services(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all healthy services.

        Returns:
            All healthy services
        """
        return {
            service_name: service
            for service_name, service in self.services.items()
            if service["status"] == "healthy"
        }

    def get_unhealthy_services(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all unhealthy services.

        Returns:
            All unhealthy services
        """
        return {
            service_name: service
            for service_name, service in self.services.items()
            if service["status"] == "unhealthy"
        }

    def get_service_for_endpoint(self, endpoint: str) -> Optional[str]:
        """
        Get the service name for an endpoint.

        Args:
            endpoint: Endpoint path

        Returns:
            Service name, or None if no service is found
        """
        for service_name, service in self.services.items():
            for service_endpoint in service["endpoints"]:
                if service_endpoint.endswith("*") and endpoint.startswith(service_endpoint[:-1]):
                    return service_name
                elif service_endpoint == endpoint:
                    return service_name

        return None