"""
Degraded Mode Service Components

This module provides functionality for services to operate in degraded mode
when dependencies are unavailable or experiencing issues.
"""

import logging
import threading
import time
import asyncio # Added for iscoroutinefunction
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, cast, Type # Added Type

# from core_foundations.events.event_schema import Event, EventType, ServiceName # Temporarily commented out
from common_lib.exceptions import ServiceUnavailableError
from core_foundations.monitoring.health_check import HealthStatus
from core_foundations.api.health_check import HealthCheck

logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class DegradedModeStrategy(str, Enum):
    """Strategies for handling degraded mode"""
    FAIL_FAST = "FAIL_FAST"  # Fail immediately if dependency is unavailable
    USE_CACHE = "USE_CACHE"  # Use cached data if available
    USE_FALLBACK = "USE_FALLBACK"  # Use fallback implementation
    REDUCED_FUNCTIONALITY = "REDUCED_FUNCTIONALITY"  # Continue with reduced functionality
    QUEUE_AND_RETRY = "QUEUE_AND_RETRY"  # Queue requests and retry when dependency recovers


class DependencyStatus:
    """Status of a service dependency"""
    
    def __init__(self, name: str):
        """
        Initialize dependency status.
        
        Args:
            name: Name of the dependency
        """
        self.name = name
        self.available = True
        self.last_check_time = datetime.utcnow()
        self.last_successful_time = datetime.utcnow()
        self.failure_count = 0
        self.success_count = 0
        self._lock = threading.RLock()
    
    def mark_success(self) -> None:
        """Mark the dependency as available"""
        with self._lock:
            self.available = True
            self.last_check_time = datetime.utcnow()
            self.last_successful_time = datetime.utcnow()
            self.failure_count = 0
            self.success_count += 1
    
    def mark_failure(self) -> None:
        """Mark the dependency as unavailable"""
        with self._lock:
            self.available = False
            self.last_check_time = datetime.utcnow()
            self.failure_count += 1
            self.success_count = 0
    
    def is_available(self) -> bool:
        """Check if the dependency is available"""
        with self._lock:
            return self.available


class DegradedModeManager:
    """
    Manager for handling degraded mode operations.
    
    This class tracks the status of service dependencies and provides
    mechanisms for operating in degraded mode when dependencies are unavailable.
    """
    
    _instance = None
    
    def __new__(cls):
        """Create a singleton instance"""
        if cls._instance is None:
            cls._instance = super(DegradedModeManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the degraded mode manager"""
        if self._initialized:
            return
        
        self._dependencies = {}
        self._fallbacks = {}
        self._degraded_flags = {}
        self._lock = threading.RLock()
        self._degraded_mode_handlers = set()
        self._initialized = True
    
    def register_dependency(self, name: str) -> None:
        """
        Register a dependency for tracking.
        
        Args:
            name: Name of the dependency
        """
        with self._lock:
            if name not in self._dependencies:
                self._dependencies[name] = DependencyStatus(name)
                logger.info(f"Registered dependency: {name}")
    
    def register_fallback(
        self,
        dependency_name: str,
        fallback_func: Callable,
        original_func: Callable
    ) -> None:
        """
        Register a fallback function for a dependency.
        
        Args:
            dependency_name: Name of the dependency
            fallback_func: Function to use when dependency is unavailable
            original_func: Original function that uses the dependency
        """
        with self._lock:
            if dependency_name not in self._dependencies:
                self.register_dependency(dependency_name)
            
            self._fallbacks[(dependency_name, original_func)] = fallback_func
            logger.info(f"Registered fallback for {dependency_name}")
    
    def get_fallback(
        self,
        dependency_name: str,
        original_func: Callable
    ) -> Optional[Callable]:
        """
        Get the fallback function for a dependency.
        
        Args:
            dependency_name: Name of the dependency
            original_func: Original function
            
        Returns:
            Fallback function or None if not registered
        """
        return self._fallbacks.get((dependency_name, original_func))
    
    def mark_dependency_available(self, name: str) -> None:
        """
        Mark a dependency as available.
        
        Args:
            name: Name of the dependency
        """
        with self._lock:
            if name in self._dependencies:
                self._dependencies[name].mark_success()
                was_degraded = self._degraded_flags.get(name, False)
                if was_degraded:
                    self._degraded_flags[name] = False
                    logger.info(f"Dependency {name} is now available, exiting degraded mode")
                    
                    # Notify handlers
                    for handler in self._degraded_mode_handlers:
                        try:
                            handler(name, False)
                        except Exception as e:
                            logger.error(f"Error in degraded mode handler: {e}")
    
    def mark_dependency_unavailable(self, name: str) -> None:
        """
        Mark a dependency as unavailable.
        
        Args:
            name: Name of the dependency
        """
        with self._lock:
            if name in self._dependencies:
                self._dependencies[name].mark_failure()
                was_degraded = self._degraded_flags.get(name, False)
                if not was_degraded:
                    self._degraded_flags[name] = True
                    logger.warning(f"Dependency {name} is unavailable, entering degraded mode")
                    
                    # Notify handlers
                    for handler in self._degraded_mode_handlers:
                        try:
                            handler(name, True)
                        except Exception as e:
                            logger.error(f"Error in degraded mode handler: {e}")
    
    def is_dependency_available(self, name: str) -> bool:
        """
        Check if a dependency is available.
        
        Args:
            name: Name of the dependency
            
        Returns:
            True if available, False otherwise
        """
        with self._lock:
            if name in self._dependencies:
                return self._dependencies[name].is_available()
            return False  # Unknown dependencies are considered unavailable
    
    def get_all_dependency_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all dependencies.
        
        Returns:
            Dictionary mapping dependency names to their status
        """
        with self._lock:
            result = {}
            for name, status in self._dependencies.items():
                result[name] = {
                    "available": status.available,
                    "last_check": status.last_check_time.isoformat(),
                    "last_success": status.last_successful_time.isoformat(),
                    "failure_count": status.failure_count,
                    "success_count": status.success_count,
                    "is_degraded": self._degraded_flags.get(name, False)
                }
            return result
    
    def add_degraded_mode_handler(self, handler: Callable[[str, bool], None]) -> None:
        """
        Add a handler to be notified when entering or exiting degraded mode.
        
        Args:
            handler: Function(dependency_name, is_degraded) to call
        """
        with self._lock:
            self._degraded_mode_handlers.add(handler)
    
    def remove_degraded_mode_handler(self, handler: Callable[[str, bool], None]) -> None:
        """
        Remove a degraded mode handler.
        
        Args:
            handler: Handler to remove
        """
        with self._lock:
            self._degraded_mode_handlers.discard(handler)


def with_degraded_mode(
    dependency_name: str,
    strategy: DegradedModeStrategy = DegradedModeStrategy.USE_FALLBACK,
    error_types: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator to apply degraded mode handling to a function.

    Args:
        dependency_name: Name of the dependency this function relies on.
        strategy: Strategy to use when the dependency is unavailable.
        error_types: Tuple of exception types that indicate dependency failure.

    Returns:
        Decorated function with degraded mode handling.
    """
    manager = DegradedModeManager()
    manager.register_dependency(dependency_name) # Ensure dependency is registered

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if manager.is_dependency_available(dependency_name):
                try:
                    result = func(*args, **kwargs)
                    # Mark success if it was previously unavailable
                    if not manager._dependencies[dependency_name].is_available():
                         manager.mark_dependency_available(dependency_name)
                    return result
                except error_types as e:
                    logger.warning(f"Call to {func.__name__} failed for dependency {dependency_name}: {e}")
                    manager.mark_dependency_unavailable(dependency_name)
                    # Fall through to degraded mode handling
                except Exception as e:
                     # Handle unexpected errors differently? Or re-raise?
                     logger.error(f"Unexpected error in {func.__name__} for dependency {dependency_name}: {e}")
                     raise # Re-raise unexpected errors

            # Dependency is unavailable or call failed, apply strategy
            logger.warning(f"Dependency {dependency_name} unavailable, applying strategy {strategy.value} for {func.__name__}")

            if strategy == DegradedModeStrategy.FAIL_FAST:
                raise ServiceUnavailableError(dependency_name)

            elif strategy == DegradedModeStrategy.USE_FALLBACK:
                fallback_func = manager.get_fallback(dependency_name, func)
                if fallback_func:
                    logger.info(f"Using fallback function for {func.__name__}")
                    return fallback_func(*args, **kwargs)
                else:
                    logger.error(f"No fallback registered for {func.__name__} with dependency {dependency_name}")
                    raise ServiceUnavailableError(dependency_name, message=f"No fallback available for {func.__name__}")

            elif strategy == DegradedModeStrategy.USE_CACHE:
                # Placeholder: Implement cache retrieval logic here
                logger.warning("USE_CACHE strategy not fully implemented")
                # Example: try cache.get(args, kwargs); if hit return else raise
                raise ServiceUnavailableError(dependency_name, message="Cache miss or cache strategy not implemented")

            elif strategy == DegradedModeStrategy.REDUCED_FUNCTIONALITY:
                # Placeholder: Function might need internal logic to handle this
                logger.warning(f"Executing {func.__name__} with reduced functionality due to unavailable {dependency_name}")
                # The decorated function itself might need to check dependency status
                # or this decorator could return a specific signal/default value.
                # For now, we'll raise an error as a default.
                raise ServiceUnavailableError(dependency_name, message="Reduced functionality not implemented, failing.")

            elif strategy == DegradedModeStrategy.QUEUE_AND_RETRY:
                # Placeholder: Implement queuing logic here
                logger.warning("QUEUE_AND_RETRY strategy not fully implemented")
                # Example: queue.enqueue(func, args, kwargs); return acknowledgement
                raise ServiceUnavailableError(dependency_name, message="Queueing strategy not implemented")

            else:
                logger.error(f"Unknown degraded mode strategy: {strategy}")
                raise ServiceUnavailableError(dependency_name)

        return cast(F, wrapper)
    return decorator


def fallback_for(dependency_name: str, original_func: Callable) -> Callable[[Callable], Callable]:
    """
    Decorator to register a fallback function for a dependency.
    
    Args:
        dependency_name: Name of the dependency
        original_func: Original function that the fallback replaces
        
    Returns:
        Decorator function
        
    Example:
        @fallback_for("feature-store", get_features)
        def get_features_fallback(symbol):
            # Fallback implementation when feature store is unavailable
    """
    def decorator(fallback_func: Callable) -> Callable:
        """Decorator function"""
        manager = DegradedModeManager()
        manager.register_fallback(dependency_name, fallback_func, original_func)
        return fallback_func
    
    return decorator


def create_health_check_dependency_monitor(
    health_check: HealthCheck,
    dependency_check_interval: float = 30.0
) -> None:
    """
    Create a dependency monitor that updates dependency status based on health checks.
    
    Args:
        health_check: Health check instance
        dependency_check_interval: Interval between checks in seconds
    """
    manager = DegradedModeManager()
    
    # Map component names to dependency names
    component_to_dependency = {
        "database": "database",
        "kafka": "event-bus",
        "feature-store-client": "feature-store",
        "data-pipeline-client": "data-pipeline",
        "analysis-engine-client": "analysis-engine",
        "ml-integration-client": "ml-integration",
        "strategy-execution-client": "strategy-execution",
        "portfolio-management-client": "portfolio-management",
        "risk-management-client": "risk-management",
        "trading-gateway-client": "trading-gateway"
    }
    
    def check_dependencies():
        """Check health of dependencies and update status"""
        health = health_check.get_health()
        
        for check in health.checks:
            # Skip checks that don't map to dependencies
            if check.component not in component_to_dependency:
                continue
            
            dependency_name = component_to_dependency[check.component]
            
            # Register dependency if not already registered
            manager.register_dependency(dependency_name)
            
            # Update dependency status
            if check.status == HealthStatus.UP:
                manager.mark_dependency_available(dependency_name)
            else:
                manager.mark_dependency_unavailable(dependency_name)
    
    # Create background thread for dependency monitoring
    stop_event = threading.Event()
    
    def monitoring_loop():
        while not stop_event.is_set():
            try:
                check_dependencies()
            except Exception as e:
                logger.error(f"Error in dependency monitoring loop: {e}")
            
            # Wait for next check
            stop_event.wait(dependency_check_interval)
    
    # Start monitoring thread
    thread = threading.Thread(
        target=monitoring_loop,
        daemon=True,
        name="dependency-monitor"
    )
    thread.start()
    
    # Return stop function
    return lambda: stop_event.set()
