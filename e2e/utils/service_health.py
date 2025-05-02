"""
Service health checking utility for E2E tests.
Validates that services are running and healthy before tests execute.
"""
import logging
import time
from typing import Dict, Any, Optional

import requests

logger = logging.getLogger(__name__)

class HealthCheckError(Exception):
    """Exception raised when a service health check fails."""
    pass

class ServiceHealthChecker:
    """
    Checks the health of services in the test environment.
    Provides methods for validating service readiness and health status.
    """
    
    def __init__(self):
        """Initialize the service health checker."""
        self.health_cache = {}  # Cache health check results
        
    def check_service_health(
        self, 
        service_name: str, 
        health_url: str, 
        timeout: int = 60,
        retry_interval: int = 2
    ) -> Dict[str, Any]:
        """
        Check if a service is healthy by querying its health endpoint.
        
        Args:
            service_name: Name of the service to check
            health_url: URL of the health endpoint
            timeout: Maximum time to wait for service to become healthy (seconds)
            retry_interval: Time between retry attempts (seconds)
            
        Returns:
            Health status information from the service
            
        Raises:
            HealthCheckError: If the service is not healthy after timeout
        """
        logger.info(f"Checking health of {service_name} at {health_url}")
        
        # Check cache first (only if we've already seen a healthy response)
        if service_name in self.health_cache:
            logger.debug(f"Using cached health status for {service_name}")
            return self.health_cache[service_name]
        
        start_time = time.time()
        last_error = None
        
        # Retry until healthy or timeout
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=retry_interval)
                
                if response.status_code == 200:
                    try:
                        health_data = response.json()
                        status = health_data.get("status", "").upper()
                        
                        if status == "UP" or status == "OK" or status == "HEALTHY":
                            # Cache successful health check
                            self.health_cache[service_name] = health_data
                            logger.info(f"Service {service_name} is healthy")
                            return health_data
                        else:
                            logger.warning(f"Service {service_name} reports status: {status}")
                            last_error = f"Unhealthy status: {status}"
                    except Exception as e:
                        logger.warning(f"Failed to parse health response from {service_name}: {e}")
                        last_error = f"Invalid health response: {str(e)}"
                else:
                    logger.warning(
                        f"Health check for {service_name} failed with status code {response.status_code}"
                    )
                    last_error = f"HTTP {response.status_code}: {response.text[:100]}"
            except requests.RequestException as e:
                logger.debug(f"Health check request to {service_name} failed: {e}")
                last_error = str(e)
                
            # Wait before retrying
            time.sleep(retry_interval)
            
        # If we get here, the service failed to become healthy within the timeout
        error_msg = f"Service {service_name} failed health check after {timeout} seconds. Last error: {last_error}"
        logger.error(error_msg)
        raise HealthCheckError(error_msg)
        
    def check_all_services(
        self, 
        service_endpoints: Dict[str, str],
        timeout_per_service: int = 30
    ) -> Dict[str, Dict[str, Any]]:
        """
        Check health of all HTTP services.
        
        Args:
            service_endpoints: Dictionary mapping service names to endpoint URLs
            timeout_per_service: Timeout per service in seconds
            
        Returns:
            Dictionary mapping service names to health status information
        """
        results = {}
        unhealthy_services = []
        
        for service_name, endpoint in service_endpoints.items():
            # Skip non-HTTP services
            if not endpoint.startswith("http"):
                logger.info(f"Skipping health check for non-HTTP service: {service_name}")
                continue
                
            health_url = f"{endpoint}/health"
            try:
                health_status = self.check_service_health(
                    service_name=service_name,
                    health_url=health_url,
                    timeout=timeout_per_service
                )
                results[service_name] = health_status
            except HealthCheckError:
                unhealthy_services.append(service_name)
                
        if unhealthy_services:
            logger.error(f"Some services are unhealthy: {', '.join(unhealthy_services)}")
            
        return results
        
    def clear_cache(self, service_name: Optional[str] = None) -> None:
        """
        Clear health check cache.
        
        Args:
            service_name: Specific service to clear, or all if None
        """
        if service_name:
            if service_name in self.health_cache:
                del self.health_cache[service_name]
        else:
            self.health_cache.clear()
