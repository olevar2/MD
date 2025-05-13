"""
Service Health Monitoring System for Forex Trading Platform

This module provides comprehensive health monitoring capabilities for
the microservices architecture of the Forex trading platform. It enables:

1. Real-time health status reporting from services
2. Automatic detection of service degradation
3. Centralized health data collection and aggregation
4. Configurable alerting based on health thresholds
5. Historical health data for trend analysis

The health monitoring system uses a combination of:
- Active health checks (polling endpoints)
- Passive monitoring (metrics collection)
- Event-based status updates (service self-reporting)
"""

import asyncio
import datetime
import json
import logging
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

try:
    import aiohttp
    import requests
except ImportError:
    raise ImportError(
        "aiohttp and requests are required for health monitoring. "
        "Install with 'pip install aiohttp requests'"
    )

from ..resilience.circuit_breaker import CircuitBreaker
from ..events.event_schema import Event, EventType, create_event
from ..events.kafka_event_bus import KafkaEventBus

# Configure logger
logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Enum representing possible health statuses for a service."""
    HEALTHY = "healthy"         # Service is fully operational
    DEGRADED = "degraded"       # Service is operational but with reduced capabilities
    UNHEALTHY = "unhealthy"     # Service is experiencing significant issues
    OFFLINE = "offline"         # Service is not responding or unreachable


class HealthMetricType(str, Enum):
    """Types of health metrics that can be collected."""
    RESPONSE_TIME = "response_time"  # API response time in ms
    CPU_USAGE = "cpu_usage"          # CPU usage percentage
    MEMORY_USAGE = "memory_usage"    # Memory usage percentage
    ERROR_RATE = "error_rate"        # Error rate percentage
    REQUESTS_PER_SECOND = "requests_per_second"  # Request throughput
    DB_QUERY_TIME = "db_query_time"  # Database query time in ms
    QUEUE_DEPTH = "queue_depth"      # Message queue depth
    ACTIVE_CONNECTIONS = "active_connections"  # Number of active connections
    SYSTEM_LOAD = "system_load"      # System load average


@dataclass
class HealthCheckConfig:
    """Configuration for a health check."""
    endpoint: str              # URL endpoint to check
    method: str = "GET"        # HTTP method to use
    timeout_seconds: float = 5.0  # Timeout for the request
    headers: Dict[str, str] = field(default_factory=dict)  # HTTP headers
    payload: Optional[Dict[str, Any]] = None  # Request payload for POST/PUT
    expected_status: int = 200  # Expected HTTP status code
    interval_seconds: int = 60  # Check interval in seconds


@dataclass
class HealthThresholds:
    """Thresholds for determining service health status."""
    response_time_warning_ms: int = 500  # Warning threshold for response time
    response_time_critical_ms: int = 2000  # Critical threshold for response time
    error_rate_warning_pct: float = 1.0  # Warning threshold for error rate
    error_rate_critical_pct: float = 5.0  # Critical threshold for error rate
    cpu_usage_warning_pct: float = 70.0  # Warning threshold for CPU usage
    cpu_usage_critical_pct: float = 90.0  # Critical threshold for CPU usage
    memory_usage_warning_pct: float = 70.0  # Warning threshold for memory usage
    memory_usage_critical_pct: float = 90.0  # Critical threshold for memory usage
    success_rate_degraded_pct: float = 95.0  # Degraded threshold for success rate
    success_rate_unhealthy_pct: float = 85.0  # Unhealthy threshold for success rate


@dataclass
class HealthMetric:
    """A single health metric data point."""
    name: str                 # Metric name
    value: float              # Metric value
    unit: str                 # Unit of measurement
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)  # Additional labels


@dataclass
class ServiceHealthData:
    """Health data for a service."""
    service_name: str                     # Name of the service
    status: HealthStatus                  # Current health status
    version: str                          # Service version
    last_check_time: datetime.datetime    # Time of the last health check
    metrics: Dict[str, List[HealthMetric]] = field(default_factory=dict)  # Metrics by type
    message: Optional[str] = None         # Optional message about health status
    
    # Time windows for metrics aggregation (in seconds)
    short_window: int = 60                # 1 minute
    medium_window: int = 300              # 5 minutes
    long_window: int = 3600               # 1 hour
    
    # Flags for advanced service states
    in_maintenance_mode: bool = False     # Service is in scheduled maintenance
    has_leader_role: bool = False         # Service is the leader in a cluster
    is_read_only: bool = False            # Service is in read-only mode
    
    # Dependencies status
    dependencies: Dict[str, HealthStatus] = field(default_factory=dict)

    def add_metric(self, metric: HealthMetric) -> None:
        """
        Add a metric data point.
        
        Args:
            metric: The metric to add
        """
        if metric.name not in self.metrics:
            self.metrics[metric.name] = []
        
        self.metrics[metric.name].append(metric)
        
        # Prune old metrics to prevent memory bloat
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(seconds=self.long_window)
        self.metrics[metric.name] = [
            m for m in self.metrics[metric.name] if m.timestamp >= cutoff
        ]

    def get_metric_average(self, metric_name: str, window_seconds: int) -> Optional[float]:
        """
        Get the average value of a metric over a time window.
        
        Args:
            metric_name: Name of the metric
            window_seconds: Time window in seconds
            
        Returns:
            Average value or None if no data available
        """
        if metric_name not in self.metrics:
            return None
            
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(seconds=window_seconds)
        values = [
            m.value for m in self.metrics[metric_name] if m.timestamp >= cutoff
        ]
        
        if not values:
            return None
            
        return statistics.mean(values)

    def get_metric_percentile(
        self, 
        metric_name: str, 
        percentile: float, 
        window_seconds: int
    ) -> Optional[float]:
        """
        Get a percentile value of a metric over a time window.
        
        Args:
            metric_name: Name of the metric
            percentile: Percentile to calculate (0.0-1.0)
            window_seconds: Time window in seconds
            
        Returns:
            Percentile value or None if no data available
        """
        if metric_name not in self.metrics:
            return None
            
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(seconds=window_seconds)
        values = sorted([
            m.value for m in self.metrics[metric_name] if m.timestamp >= cutoff
        ])
        
        if not values:
            return None
            
        index = int(len(values) * percentile)
        return values[min(index, len(values) - 1)]


class HealthMonitor:
    """
    Service health monitoring system.
    
    This class handles health checks, metrics collection, and status reporting
    for services in the Forex trading platform.
    """
    
    def __init__(
        self,
        service_name: str,
        version: str,
        health_check_endpoint: str = "/health",
        metrics_endpoint: str = "/metrics",
        event_bus: Optional[KafkaEventBus] = None
    ):
        """
        Initialize the health monitor for a service.
        
        Args:
            service_name: Name of the service being monitored
            version: Version of the service
            health_check_endpoint: Endpoint for health checks
            metrics_endpoint: Endpoint for metrics collection
            event_bus: Optional event bus for publishing health events
        """
        self.service_name = service_name
        self.version = version
        self.health_check_endpoint = health_check_endpoint
        self.metrics_endpoint = metrics_endpoint
        self.event_bus = event_bus
        
        # Initialize health data
        self.health_data = ServiceHealthData(
            service_name=service_name,
            status=HealthStatus.HEALTHY,  # Start with assumption of health
            version=version,
            last_check_time=datetime.datetime.utcnow()
        )
        
        # Health check configurations for dependencies
        self.health_check_configs: Dict[str, HealthCheckConfig] = {}
        
        # Thresholds for health status determination
        self.thresholds = HealthThresholds()
        
        # Background check thread
        self._check_thread = None
        self._stop_event = threading.Event()
        
        # Circuit breaker for health check resilience
        self.circuit_breaker = CircuitBreaker(
            name=f"{service_name}_health_monitor",
            failure_threshold=3,
            recovery_timeout=30
        )

    def add_dependency_check(
        self, 
        dependency_name: str, 
        config: HealthCheckConfig
    ) -> None:
        """
        Add a health check configuration for a dependency.
        
        Args:
            dependency_name: Name of the dependency
            config: Health check configuration
        """
        self.health_check_configs[dependency_name] = config
        self.health_data.dependencies[dependency_name] = HealthStatus.UNKNOWN

    def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._check_thread is not None and self._check_thread.is_alive():
            logger.warning("Health monitoring already started")
            return
            
        self._stop_event.clear()
        self._check_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._check_thread.start()
        logger.info(f"Health monitoring started for {self.service_name}")

    def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        if self._check_thread is None or not self._check_thread.is_alive():
            logger.warning("Health monitoring not running")
            return
            
        self._stop_event.set()
        self._check_thread.join(timeout=5.0)
        logger.info(f"Health monitoring stopped for {self.service_name}")

    def _monitoring_loop(self) -> None:
        """Background loop for continuous health monitoring."""
        while not self._stop_event.is_set():
            try:
                # Check dependencies
                for dep_name, config in self.health_check_configs.items():
                    try:
                        status = self._check_dependency(dep_name, config)
                        self.health_data.dependencies[dep_name] = status
                    except Exception as e:
                        logger.warning(f"Failed to check dependency {dep_name}: {e}")
                        self.health_data.dependencies[dep_name] = HealthStatus.UNKNOWN
                
                # Update overall health status
                self._update_health_status()
                
                # Report health status if event bus is available
                if self.event_bus:
                    self._report_health_status()
                
                # Wait for next check interval or until stop is requested
                self._stop_event.wait(10.0)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(5.0)  # Wait a bit before retrying

    def _check_dependency(
        self, 
        dependency_name: str, 
        config: HealthCheckConfig
    ) -> HealthStatus:
        """
        Check the health of a dependency.
        
        Args:
            dependency_name: Name of the dependency
            config: Health check configuration
            
        Returns:
            HealthStatus of the dependency
        """
        try:
            # Use the circuit breaker to prevent excessive checks when failing
            if not self.circuit_breaker.allow_request():
                logger.warning(
                    f"Circuit breaker preventing health check for {dependency_name}"
                )
                return HealthStatus.UNKNOWN
                
            start_time = time.time()
            
            # Make the request
            response = requests.request(
                method=config.method,
                url=config.endpoint,
                headers=config.headers,
                json=config.payload,
                timeout=config.timeout_seconds
            )
            
            # Record response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Add response time metric
            self.add_metric(
                metric_name="response_time",
                value=response_time_ms,
                unit="ms",
                labels={"dependency": dependency_name}
            )
            
            # Check if response is as expected
            if response.status_code == config.expected_status:
                self.circuit_breaker.record_success()
                
                # Parse response for more detailed health status if available
                try:
                    health_data = response.json()
                    if "status" in health_data:
                        status_str = health_data["status"].lower()
                        if status_str in [status.value for status in HealthStatus]:
                            return HealthStatus(status_str)
                except (ValueError, KeyError):
                    pass
                    
                # Determine status based on response time
                if response_time_ms > self.thresholds.response_time_critical_ms:
                    return HealthStatus.DEGRADED
                return HealthStatus.HEALTHY
            else:
                self.circuit_breaker.record_failure()
                return HealthStatus.UNHEALTHY
                
        except requests.RequestException:
            self.circuit_breaker.record_failure()
            return HealthStatus.OFFLINE
        except Exception as e:
            logger.error(f"Error checking {dependency_name}: {e}")
            self.circuit_breaker.record_failure()
            return HealthStatus.UNKNOWN

    def _update_health_status(self) -> None:
        """Update the overall health status based on metrics and dependency statuses."""
        # Check if any critical metrics are breached
        status = HealthStatus.HEALTHY
        messages = []
        
        # Check response time (if available)
        avg_response_time = self.health_data.get_metric_average(
            "response_time", 
            self.health_data.short_window
        )
        if avg_response_time is not None:
            if avg_response_time > self.thresholds.response_time_critical_ms:
                status = HealthStatus.DEGRADED
                messages.append(f"Response time critical: {avg_response_time:.2f}ms")
            elif avg_response_time > self.thresholds.response_time_warning_ms:
                status = max(status, HealthStatus.DEGRADED)
                messages.append(f"Response time elevated: {avg_response_time:.2f}ms")
        
        # Check error rate (if available)
        error_rate = self.health_data.get_metric_average(
            "error_rate", 
            self.health_data.short_window
        )
        if error_rate is not None:
            if error_rate > self.thresholds.error_rate_critical_pct:
                status = HealthStatus.UNHEALTHY
                messages.append(f"Error rate critical: {error_rate:.2f}%")
            elif error_rate > self.thresholds.error_rate_warning_pct:
                status = max(status, HealthStatus.DEGRADED)
                messages.append(f"Error rate elevated: {error_rate:.2f}%")
        
        # Check dependency statuses
        unhealthy_deps = []
        degraded_deps = []
        
        for dep_name, dep_status in self.health_data.dependencies.items():
            if dep_status == HealthStatus.UNHEALTHY:
                unhealthy_deps.append(dep_name)
            elif dep_status == HealthStatus.DEGRADED:
                degraded_deps.append(dep_name)
        
        if unhealthy_deps:
            status = max(status, HealthStatus.DEGRADED)
            messages.append(f"Unhealthy dependencies: {', '.join(unhealthy_deps)}")
            
        if degraded_deps:
            status = max(status, HealthStatus.DEGRADED)
            messages.append(f"Degraded dependencies: {', '.join(degraded_deps)}")
        
        # Update health data
        self.health_data.status = status
        self.health_data.last_check_time = datetime.datetime.utcnow()
        self.health_data.message = "; ".join(messages) if messages else None

    def _report_health_status(self) -> None:
        """Report health status to the event bus."""
        if not self.event_bus:
            return
            
        # Only report if status has changed or it's been a while
        try:
            event = create_event(
                event_type=EventType.SERVICE_HEALTH_CHANGED,
                source_service=self.service_name,
                data={
                    "status": self.health_data.status.value,
                    "service_name": self.service_name,
                    "version": self.version,
                    "message": self.health_data.message,
                    "timestamp": self.health_data.last_check_time.isoformat(),
                    "dependencies": {
                        name: status.value 
                        for name, status in self.health_data.dependencies.items()
                    }
                }
            )
            
            self.event_bus.publish(event)
        except Exception as e:
            logger.error(f"Failed to report health status: {e}")

    def add_metric(
        self, 
        metric_name: str, 
        value: float, 
        unit: str,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Add a metric data point.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            labels: Additional labels for the metric
        """
        metric = HealthMetric(
            name=metric_name,
            value=value,
            unit=unit,
            labels=labels or {}
        )
        
        self.health_data.add_metric(metric)

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get the current health status.
        
        Returns:
            Dictionary with health status information
        """
        return {
            "service_name": self.health_data.service_name,
            "status": self.health_data.status.value,
            "version": self.health_data.version,
            "message": self.health_data.message,
            "last_check_time": self.health_data.last_check_time.isoformat(),
            "dependencies": {
                name: status.value 
                for name, status in self.health_data.dependencies.items()
            },
            "metrics": {
                name: [asdict(m) for m in metrics[-5:]]  # Last 5 readings
                for name, metrics in self.health_data.metrics.items()
            }
        }

    def set_maintenance_mode(self, enabled: bool) -> None:
        """
        Set maintenance mode status.
        
        Args:
            enabled: True to enable maintenance mode, False to disable
        """
        self.health_data.in_maintenance_mode = enabled
        if enabled:
            logger.info(f"Maintenance mode enabled for {self.service_name}")
        else:
            logger.info(f"Maintenance mode disabled for {self.service_name}")
            
        # Report status change
        self._report_health_status()


class HealthRegistry:
    """
    Registry for tracking health status of multiple services.
    
    This central registry collects and aggregates health data from
    all services in the system, providing a comprehensive view.
    """
    
    def __init__(self, service_name: str, event_bus: KafkaEventBus):
        """
        Initialize the health registry.
        
        Args:
            service_name: Name of the service running this registry
            event_bus: Event bus for receiving health events
        """
        self.service_name = service_name
        self.event_bus = event_bus
        
        # Health data by service name
        self.services: Dict[str, Dict[str, Any]] = {}
        
        # Last time a service reported health
        self.last_report_time: Dict[str, datetime.datetime] = {}
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Subscribe to health events
        self.event_bus.subscribe(
            [EventType.SERVICE_HEALTH_CHANGED],
            self._handle_health_event
        )
        self.event_bus.start_consuming(blocking=False)

    def _handle_health_event(self, event: Event) -> None:
        """
        Handle a health status event.
        
        Args:
            event: The health status event
        """
        if event.event_type != EventType.SERVICE_HEALTH_CHANGED:
            return
            
        service_name = event.data.get("service_name")
        if not service_name:
            logger.warning("Received health event without service_name")
            return
            
        with self._lock:
            # Update service health data
            self.services[service_name] = event.data
            self.last_report_time[service_name] = datetime.datetime.utcnow()

    def get_service_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the health status of a specific service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Health status or None if not found
        """
        with self._lock:
            return self.services.get(service_name)

    def get_all_services_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the health status of all services.
        
        Returns:
            Dictionary mapping service names to health status
        """
        with self._lock:
            return dict(self.services)

    def get_system_health_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the health of the entire system.
        
        Returns:
            Dictionary with system health summary
        """
        with self._lock:
            status_counts = {
                HealthStatus.HEALTHY.value: 0,
                HealthStatus.DEGRADED.value: 0,
                HealthStatus.UNHEALTHY.value: 0,
                HealthStatus.OFFLINE.value: 0
            }
            
            current_time = datetime.datetime.utcnow()
            total_services = len(self.services)
            reporting_services = 0
            
            for service_name, health_data in self.services.items():
                status = health_data.get("status", "unknown")
                if status in status_counts:
                    status_counts[status] += 1
                
                # Check if service is still reporting
                last_report = self.last_report_time.get(service_name)
                if last_report:
                    age = (current_time - last_report).total_seconds()
                    if age < 300:  # Considered reporting if within 5 minutes
                        reporting_services += 1
            
            # Overall system status
            system_status = HealthStatus.HEALTHY.value
            if status_counts[HealthStatus.UNHEALTHY.value] > 0:
                system_status = HealthStatus.DEGRADED.value
            if status_counts[HealthStatus.UNHEALTHY.value] >= total_services * 0.25:
                system_status = HealthStatus.UNHEALTHY.value
                
            return {
                "timestamp": current_time.isoformat(),
                "system_status": system_status,
                "total_services": total_services,
                "reporting_services": reporting_services,
                "status_counts": status_counts
            }


# class HealthCheck:
    """
    HealthCheck class.
    
    Attributes:
        Add attributes here
    """
 # This was the incomplete definition
#     """
#     Health check implementation for exposing health status via HTTP.
#     
#     This class provides a standard health check endpoint that can be
#     exposed by services to report their health status.
#     """
#     
#     # The full implementation is now in core_foundations.api.health_check
#     # This placeholder can be removed if api.health_check is the intended location.
#     # If this was meant to be a different class, it needs full implementation.
#     pass # Remove this pass if api.health_check.HealthCheck is the correct one.
