"""
Health Check Module for Forex Trading Platform

This module provides standardized health check functionality for all services.
It includes:
- Configurable health check endpoints
- Component-level health monitoring
- Degraded mode operation handling
- Health status reporting to the event bus
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable

from pydantic import BaseModel, Field

from ..events.kafka_event_bus import KafkaEventBus
from ..events.event_schema import EventType, Event

logger = logging.getLogger(__name__)

class HealthStatus(str, Enum):
    """Health status values for services and components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentHealth(BaseModel):
    """Health information for a specific component."""
    name: str = Field(..., description="Name of the component")
    status: HealthStatus = Field(HealthStatus.UNKNOWN, description="Health status of the component")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details about the component's health")
    last_check_timestamp: Optional[float] = Field(None, description="Timestamp of the last health check")


class ServiceHealth(BaseModel):
    """Health information for the entire service."""
    service_name: str = Field(..., description="Name of the service")
    version: str = Field(..., description="Service version")
    status: HealthStatus = Field(HealthStatus.UNKNOWN, description="Overall health status of the service")
    components: Dict[str, ComponentHealth] = Field(
        default_factory=dict,
        description="Health status of individual components"
    )
    uptime_seconds: float = Field(0.0, description="Service uptime in seconds")
    deployment_environment: str = Field("development", description="Deployment environment")
    additional_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional service-specific information"
    )

    def get_overall_status(self) -> HealthStatus:
        """Calculate the overall health status based on component statuses."""
        if not self.components:
            return HealthStatus.UNKNOWN
            
        statuses = [component.status for component in self.components.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN


class HealthCheckManager:
    """
    Central manager for service health checks.
    
    This class handles:
    - Tracking component health
    - Evaluating overall service health
    - Publishing health events
    - Managing degraded mode operations
    """
    
    def __init__(
        self, 
        service_name: str,
        version: str,
        event_bus: Optional[KafkaEventBus] = None,
        deployment_environment: str = "development"
    ):
        """
        Initialize the health check manager.
        
        Args:
            service_name: Name of the service
            version: Service version string
            event_bus: Optional event bus for publishing health events
            deployment_environment: Deployment environment (e.g., "development", "production")
        """
        self.service_health = ServiceHealth(
            service_name=service_name,
            version=version,
            deployment_environment=deployment_environment
        )
        self.event_bus = event_bus
        self.start_time = self._get_current_time()
        self._degraded_mode_strategies: Dict[str, Callable] = {}
        self._report_health_interval_seconds = 60.0  # Report health every minute by default
        self._last_health_report_time = 0.0
        
    def _get_current_time(self) -> float:
        """Get current time in seconds since epoch."""
        import time
        return time.time()
        
    def register_component(
        self, 
        component_name: str, 
        check_function: Callable[[], Tuple[HealthStatus, Dict[str, Any]]]
    ) -> None:
        """
        Register a component with its health check function.
        
        Args:
            component_name: Name of the component
            check_function: Function that returns health status and details
        """
        self.service_health.components[component_name] = ComponentHealth(
            name=component_name,
            status=HealthStatus.UNKNOWN
        )
        
    def register_degraded_mode_strategy(
        self,
        component_name: str,
        strategy_function: Callable[[], None]
    ) -> None:
        """
        Register a strategy to handle degraded mode for a component.
        
        Args:
            component_name: Name of the component
            strategy_function: Function to call when the component is degraded
        """
        if component_name not in self.service_health.components:
            raise ValueError(f"Component {component_name} not registered")
            
        self._degraded_mode_strategies[component_name] = strategy_function
        
    def update_component_health(
        self,
        component_name: str,
        status: HealthStatus,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update the health status of a component.
        
        Args:
            component_name: Name of the component
            status: Health status
            details: Optional details about the health status
        """
        if component_name not in self.service_health.components:
            raise ValueError(f"Component {component_name} not registered")
            
        self.service_health.components[component_name].status = status
        
        if details:
            self.service_health.components[component_name].details = details
            
        self.service_health.components[component_name].last_check_timestamp = self._get_current_time()
        
        # Update overall service health
        self.service_health.status = self.service_health.get_overall_status()
        
        # Handle degraded or unhealthy component
        if status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY):
            self._handle_component_degradation(component_name)
            
        # Consider reporting health status
        self._maybe_report_health_status()
            
    def _handle_component_degradation(self, component_name: str) -> None:
        """
        Handle degraded or unhealthy component.
        
        Args:
            component_name: Name of the degraded component
        """
        if component_name in self._degraded_mode_strategies:
            try:
                logger.info(f"Activating degraded mode strategy for {component_name}")
                self._degraded_mode_strategies[component_name]()
            except Exception as e:
                logger.error(f"Error in degraded mode strategy for {component_name}: {e}")
                
    def _maybe_report_health_status(self) -> None:
        """Report health status if the reporting interval has elapsed."""
        current_time = self._get_current_time()
        
        # Check if it's time to report health status
        if (self.event_bus is not None and 
            current_time - self._last_health_report_time > self._report_health_interval_seconds):
            
            self._report_health_status()
            self._last_health_report_time = current_time
            
    def _report_health_status(self) -> None:
        """Report health status to the event bus."""
        if self.event_bus is None:
            return
            
        try:
            # Update uptime
            self.service_health.uptime_seconds = self._get_current_time() - self.start_time
            
            # Publish health event
            health_data = self.service_health.dict()
            
            self.event_bus.publish(
                event=Event(
                    event_type=EventType.SYSTEM_HEALTH_REPORT,
                    source_service=self.service_health.service_name,
                    data=health_data
                )
            )
            
            logger.debug(f"Published health report for {self.service_health.service_name}")
        except Exception as e:
            logger.error(f"Failed to report health status: {e}")
            
    def check_all_components(self) -> None:
        """Run health checks for all registered components."""
        # Update uptime first
        self.service_health.uptime_seconds = self._get_current_time() - self.start_time
        
        # Check each component
        for component_name, component in self.service_health.components.items():
            try:
                # Get the check function for this component
                check_function = getattr(self, f"check_{component_name}", None)
                
                if check_function and callable(check_function):
                    status, details = check_function()
                    self.update_component_health(
                        component_name=component_name,
                        status=status,
                        details=details
                    )
            except Exception as e:
                logger.error(f"Error checking health of {component_name}: {e}")
                self.update_component_health(
                    component_name=component_name,
                    status=HealthStatus.UNHEALTHY,
                    details={"error": str(e)}
                )
                
        # Update overall status
        self.service_health.status = self.service_health.get_overall_status()
        
        # Report health status
        self._report_health_status()
        
    def get_health_data(self) -> Dict[str, Any]:
        """
        Get complete health data for API responses.
        
        Returns:
            Dict: Complete health data
        """
        # Update uptime
        self.service_health.uptime_seconds = self._get_current_time() - self.start_time
        
        return self.service_health.dict()

def create_database_health_check(db_session) -> Callable[[], Tuple[HealthStatus, Dict[str, Any]]]:
    """
    Create a health check function for a database connection.
    
    Args:
        db_session: Database session object
        
    Returns:
        Function that performs the health check
    """
    async def check_database() -> Tuple[HealthStatus, Dict[str, Any]]:
        """Check database connectivity."""
        try:
            # Execute a simple query to check connection
            await db_session.execute("SELECT 1")
            
            return HealthStatus.HEALTHY, {
                "message": "Database connection is healthy",
                "latency_ms": 0  # You could measure this more accurately
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return HealthStatus.UNHEALTHY, {
                "message": f"Database connection failed: {str(e)}",
                "error": str(e)
            }
            
    return check_database

def create_kafka_health_check(kafka_client: KafkaEventBus) -> Callable[[], Tuple[HealthStatus, Dict[str, Any]]]:
    """
    Create a health check function for a Kafka connection.
    
    Args:
        kafka_client: KafkaEventBus instance
        
    Returns:
        Function that performs the health check
    """
    def check_kafka() -> Tuple[HealthStatus, Dict[str, Any]]:
        """Check Kafka connectivity."""
        try:
            # A simple check that tries to access the Kafka admin client
            if hasattr(kafka_client, '_admin_client') and kafka_client._admin_client:
                # If the client exists, we can consider it healthy for now
                # A more thorough check would try to publish a message
                return HealthStatus.HEALTHY, {
                    "message": "Kafka connection is available",
                    "broker_count": 1  # This is a placeholder; ideally get actual count
                }
            else:
                return HealthStatus.DEGRADED, {
                    "message": "Kafka admin client is not available",
                    "suggestion": "Check Kafka bootstrap servers configuration"
                }
        except Exception as e:
            logger.error(f"Kafka health check failed: {e}")
            return HealthStatus.UNHEALTHY, {
                "message": f"Kafka connection failed: {str(e)}",
                "error": str(e)
            }
            
    return check_kafka
