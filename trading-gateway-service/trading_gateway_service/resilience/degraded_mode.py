"""
Trading Gateway Service Degraded Mode Strategies for Forex Trading Platform

This module implements degraded mode strategies for the Trading Gateway Service,
allowing it to continue functioning with reduced capabilities when facing
issues with dependencies or resource constraints. These strategies are essential
for maintaining critical trading operations during partial system failures.

Key capabilities:
1. Detecting when to enter degraded modes
2. Implementing fallback behaviors for critical operations
3. Prioritizing operations during resource constraints
4. Graceful service degradation with clear client communication
5. Automatic recovery when conditions improve
"""

import datetime
import enum
import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, cast

from ..events.event_schema import Event, EventType, EventPriority, create_event
from ..events.kafka_event_bus import KafkaEventBus
from ..monitoring.health_monitoring import HealthStatus
from ..resilience.circuit_breaker import CircuitBreaker, CircuitBreakerException

# Configure logger
logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar('T')


class DegradationLevel(enum.IntEnum):
    """Levels of service degradation, from normal to severely degraded."""
    NORMAL = 0          # Full functionality
    LIGHT = 1           # Slight degradation, all features work but some might be slower
    MODERATE = 2        # Moderate degradation, non-critical features disabled
    SEVERE = 3          # Severe degradation, only critical operations available
    CRITICAL = 4        # Critical degradation, emergency operations only


class DegradationReason(str, Enum):
    """Reasons for entering a degraded mode."""
    BROKER_CONNECTIVITY = "broker_connectivity"  # Issues with broker connection
    MARKET_DATA_QUALITY = "market_data_quality"  # Poor quality or stale market data
    LATENCY = "latency"                          # High latency to trading venues
    RATE_LIMITING = "rate_limiting"              # Being rate-limited by broker/exchange
    INTERNAL_ERROR = "internal_error"            # Internal service errors
    DEPENDENCY_FAILURE = "dependency_failure"    # Failure of a dependent service
    RESOURCE_CONSTRAINT = "resource_constraint"  # Resource constraints (CPU, memory, etc.)
    SCHEDULED_MAINTENANCE = "scheduled_maintenance"  # Planned maintenance
    ADMINISTRATIVE = "administrative"            # Manual administrative action


@dataclass
class DegradationState:
    """Current state of service degradation."""
    level: DegradationLevel = DegradationLevel.NORMAL
    reasons: Set[DegradationReason] = field(default_factory=set)
    start_time: Optional[datetime.datetime] = None
    last_update_time: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    message: Optional[str] = None


@dataclass
class DegradationThresholds:
    """Thresholds for determining when to enter different degradation levels."""
    # Broker connectivity thresholds
    broker_error_rate_light: float = 0.01  # 1% error rate
    broker_error_rate_moderate: float = 0.05  # 5% error rate
    broker_error_rate_severe: float = 0.15  # 15% error rate
    broker_error_rate_critical: float = 0.30  # 30% error rate
    
    # Market data quality thresholds
    market_data_staleness_light_sec: float = 5.0  # Data 5 seconds stale
    market_data_staleness_moderate_sec: float = 15.0  # Data 15 seconds stale
    market_data_staleness_severe_sec: float = 30.0  # Data 30 seconds stale
    market_data_staleness_critical_sec: float = 60.0  # Data 60 seconds stale
    
    # Latency thresholds
    latency_light_ms: float = 200.0  # 200ms latency
    latency_moderate_ms: float = 500.0  # 500ms latency
    latency_severe_ms: float = 1000.0  # 1 second latency
    latency_critical_ms: float = 3000.0  # 3 seconds latency
    
    # Rate limiting thresholds
    rate_limit_light_pct: float = 70.0  # 70% of rate limit
    rate_limit_moderate_pct: float = 85.0  # 85% of rate limit
    rate_limit_severe_pct: float = 95.0  # 95% of rate limit
    rate_limit_critical_pct: float = 100.0  # 100% of rate limit (being throttled)


class FallbackOperation(str, Enum):
    """Types of fallback operations available in degraded mode."""
    CACHED_MARKET_DATA = "cached_market_data"  # Use cached market data
    SIMULATED_EXECUTION = "simulated_execution"  # Simulate order execution
    DELAYED_EXECUTION = "delayed_execution"  # Queue orders for later execution
    REDUCED_REPORTING = "reduced_reporting"  # Reduce granularity of reporting
    SIMPLIFIED_VALIDATION = "simplified_validation"  # Simplify order validation
    BATCH_PROCESSING = "batch_processing"  # Process orders in batches
    LOCAL_RISK_CHECK = "local_risk_check"  # Use local risk checks instead of service
    EMERGENCY_CLOSE_ONLY = "emergency_close_only"  # Only allow closing positions


@dataclass
class FallbackStrategy:
    """Strategy for handling operations in a degraded mode."""
    operation: FallbackOperation  # Type of fallback operation
    min_degradation_level: DegradationLevel  # Minimum degradation level to activate
    applicable_reasons: Set[DegradationReason]  # Reasons this strategy applies to
    description: str  # Human-readable description
    timeout_sec: float = 3600.0  # How long this strategy can be used (seconds)


class OperationPriority(enum.IntEnum):
    """Priority levels for operations during degraded mode."""
    CRITICAL = 0    # Must always be processed (e.g., emergency close)
    HIGH = 1        # Should be processed if possible (e.g., client close order)
    MEDIUM = 2      # Process if resources allow (e.g., client open order)
    LOW = 3         # Process only in good conditions (e.g., scheduled reports)
    BACKGROUND = 4  # Process only in normal conditions (e.g., analytics)


class TradingGatewayDegradedMode:
    """
    Manager for degraded mode operations in the Trading Gateway Service.
    
    This class handles the detection, entry, management, and exit of degraded
    operational modes for the trading gateway service, implementing appropriate
    fallback strategies based on the nature of the degradation.
    """
    
    def __init__(
        self,
        service_name: str,
        event_bus: Optional[KafkaEventBus] = None,
        thresholds: Optional[DegradationThresholds] = None
    ):
        """
        Initialize the degraded mode manager.
        
        Args:
            service_name: Name of the trading gateway service instance
            event_bus: Optional event bus for status reporting
            thresholds: Optional custom thresholds
        """
        self.service_name = service_name
        self.event_bus = event_bus
        self.thresholds = thresholds or DegradationThresholds()
        
        # Track current state
        self.state = DegradationState()
        
        # Set up fallback strategies
        self.fallback_strategies: List[FallbackStrategy] = self._initialize_strategies()
        
        # Keep track of active fallbacks
        self.active_fallbacks: Set[FallbackOperation] = set()
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
    def _initialize_strategies(self) -> List[FallbackStrategy]:
        """
        Initialize the list of available fallback strategies.
        
        Returns:
            List of fallback strategies
        """
        return [
            FallbackStrategy(
                operation=FallbackOperation.CACHED_MARKET_DATA,
                min_degradation_level=DegradationLevel.LIGHT,
                applicable_reasons={
                    DegradationReason.MARKET_DATA_QUALITY,
                    DegradationReason.DEPENDENCY_FAILURE,
                    DegradationReason.LATENCY
                },
                description="Use cached market data instead of real-time market data"
            ),
            FallbackStrategy(
                operation=FallbackOperation.SIMPLIFIED_VALIDATION,
                min_degradation_level=DegradationLevel.LIGHT,
                applicable_reasons={
                    DegradationReason.LATENCY,
                    DegradationReason.RESOURCE_CONSTRAINT,
                    DegradationReason.DEPENDENCY_FAILURE
                },
                description="Use simplified order validation"
            ),
            FallbackStrategy(
                operation=FallbackOperation.LOCAL_RISK_CHECK,
                min_degradation_level=DegradationLevel.MODERATE,
                applicable_reasons={
                    DegradationReason.DEPENDENCY_FAILURE,
                    DegradationReason.LATENCY
                },
                description="Use local risk checks instead of risk management service"
            ),
            FallbackStrategy(
                operation=FallbackOperation.REDUCED_REPORTING,
                min_degradation_level=DegradationLevel.MODERATE,
                applicable_reasons={
                    DegradationReason.RESOURCE_CONSTRAINT,
                    DegradationReason.LATENCY,
                    DegradationReason.RATE_LIMITING
                },
                description="Reduce frequency and detail of reporting"
            ),
            FallbackStrategy(
                operation=FallbackOperation.BATCH_PROCESSING,
                min_degradation_level=DegradationLevel.MODERATE,
                applicable_reasons={
                    DegradationReason.RATE_LIMITING,
                    DegradationReason.RESOURCE_CONSTRAINT
                },
                description="Process orders in batches"
            ),
            FallbackStrategy(
                operation=FallbackOperation.DELAYED_EXECUTION,
                min_degradation_level=DegradationLevel.SEVERE,
                applicable_reasons={
                    DegradationReason.BROKER_CONNECTIVITY,
                    DegradationReason.RATE_LIMITING,
                    DegradationReason.INTERNAL_ERROR
                },
                description="Queue orders for later execution"
            ),
            FallbackStrategy(
                operation=FallbackOperation.SIMULATED_EXECUTION,
                min_degradation_level=DegradationLevel.SEVERE,
                applicable_reasons={
                    DegradationReason.BROKER_CONNECTIVITY,
                    DegradationReason.SCHEDULED_MAINTENANCE
                },
                description="Simulate order execution"
            ),
            FallbackStrategy(
                operation=FallbackOperation.EMERGENCY_CLOSE_ONLY,
                min_degradation_level=DegradationLevel.CRITICAL,
                applicable_reasons={
                    DegradationReason.BROKER_CONNECTIVITY,
                    DegradationReason.INTERNAL_ERROR,
                    DegradationReason.ADMINISTRATIVE
                },
                description="Only allow emergency position closing"
            )
        ]
        
    def enter_degraded_mode(
        self, 
        level: DegradationLevel,
        reason: DegradationReason,
        message: Optional[str] = None
    ) -> None:
        """
        Explicitly enter a degraded mode with the specified level and reason.
        
        Args:
            level: The degradation level to enter
            reason: The reason for entering degraded mode
            message: Optional explanatory message
        """
        with self._lock:
            # Update state
            self.state.level = max(self.state.level, level)
            self.state.reasons.add(reason)
            self.state.last_update_time = datetime.datetime.utcnow()
            
            if message:
                self.state.message = message
                
            # Set start time if first degradation
            if self.state.level > DegradationLevel.NORMAL and self.state.start_time is None:
                self.state.start_time = datetime.datetime.utcnow()
                
            # Activate appropriate fallback strategies
            self._update_active_fallbacks()
            
            # Report status change
            self._report_status_change()
            
            logger.warning(
                f"Entered degraded mode: level={level.name}, reason={reason.value}, "
                f"message={message}"
            )
            
    def exit_degraded_mode(self, reason: DegradationReason) -> None:
        """
        Exit from a degraded mode for the specified reason.
        
        Args:
            reason: The reason that no longer applies
        """
        with self._lock:
            if reason in self.state.reasons:
                self.state.reasons.remove(reason)
                self.state.last_update_time = datetime.datetime.utcnow()
                
                # Recalculate degradation level based on remaining reasons
                if not self.state.reasons:
                    # No more reasons, return to normal
                    self.state.level = DegradationLevel.NORMAL
                    self.state.start_time = None
                    self.state.message = None
                else:
                    # Recalculate level based on remaining reasons
                    # This is a simplified approach - in a real implementation,
                    # you would have logic to determine the level based on active reasons
                    pass
                    
                # Update active fallbacks
                self._update_active_fallbacks()
                
                # Report status change
                self._report_status_change()
                
                logger.info(
                    f"Exited degraded mode for reason: {reason.value}, "
                    f"current level: {self.state.level.name}"
                )
                
    def detect_broker_connectivity_issues(self, error_rate: float) -> None:
        """
        Detect broker connectivity issues based on error rate.
        
        Args:
            error_rate: Current error rate for broker operations
        """
        with self._lock:
            if error_rate >= self.thresholds.broker_error_rate_critical:
                self.enter_degraded_mode(
                    DegradationLevel.CRITICAL,
                    DegradationReason.BROKER_CONNECTIVITY,
                    f"Critical broker connectivity issues: {error_rate:.2%} error rate"
                )
            elif error_rate >= self.thresholds.broker_error_rate_severe:
                self.enter_degraded_mode(
                    DegradationLevel.SEVERE,
                    DegradationReason.BROKER_CONNECTIVITY,
                    f"Severe broker connectivity issues: {error_rate:.2%} error rate"
                )
            elif error_rate >= self.thresholds.broker_error_rate_moderate:
                self.enter_degraded_mode(
                    DegradationLevel.MODERATE,
                    DegradationReason.BROKER_CONNECTIVITY,
                    f"Moderate broker connectivity issues: {error_rate:.2%} error rate"
                )
            elif error_rate >= self.thresholds.broker_error_rate_light:
                self.enter_degraded_mode(
                    DegradationLevel.LIGHT,
                    DegradationReason.BROKER_CONNECTIVITY,
                    f"Light broker connectivity issues: {error_rate:.2%} error rate"
                )
            elif DegradationReason.BROKER_CONNECTIVITY in self.state.reasons:
                # Error rate is now below threshold, exit degraded mode
                self.exit_degraded_mode(DegradationReason.BROKER_CONNECTIVITY)
                
    def detect_market_data_quality_issues(self, staleness_seconds: float) -> None:
        """
        Detect market data quality issues based on data staleness.
        
        Args:
            staleness_seconds: How stale the market data is, in seconds
        """
        with self._lock:
            if staleness_seconds >= self.thresholds.market_data_staleness_critical_sec:
                self.enter_degraded_mode(
                    DegradationLevel.CRITICAL,
                    DegradationReason.MARKET_DATA_QUALITY,
                    f"Critical market data quality issues: {staleness_seconds:.1f}s stale"
                )
            elif staleness_seconds >= self.thresholds.market_data_staleness_severe_sec:
                self.enter_degraded_mode(
                    DegradationLevel.SEVERE,
                    DegradationReason.MARKET_DATA_QUALITY,
                    f"Severe market data quality issues: {staleness_seconds:.1f}s stale"
                )
            elif staleness_seconds >= self.thresholds.market_data_staleness_moderate_sec:
                self.enter_degraded_mode(
                    DegradationLevel.MODERATE,
                    DegradationReason.MARKET_DATA_QUALITY,
                    f"Moderate market data quality issues: {staleness_seconds:.1f}s stale"
                )
            elif staleness_seconds >= self.thresholds.market_data_staleness_light_sec:
                self.enter_degraded_mode(
                    DegradationLevel.LIGHT,
                    DegradationReason.MARKET_DATA_QUALITY,
                    f"Light market data quality issues: {staleness_seconds:.1f}s stale"
                )
            elif DegradationReason.MARKET_DATA_QUALITY in self.state.reasons:
                # Data staleness is now below threshold, exit degraded mode
                self.exit_degraded_mode(DegradationReason.MARKET_DATA_QUALITY)
                
    def detect_service_dependency_issues(
        self, 
        dependency_name: str,
        status: HealthStatus
    ) -> None:
        """
        Detect issues with service dependencies based on health status.
        
        Args:
            dependency_name: Name of the dependency service
            status: Health status of the dependency
        """
        with self._lock:
            if status == HealthStatus.OFFLINE:
                self.enter_degraded_mode(
                    DegradationLevel.SEVERE,
                    DegradationReason.DEPENDENCY_FAILURE,
                    f"Dependency {dependency_name} is OFFLINE"
                )
            elif status == HealthStatus.UNHEALTHY:
                self.enter_degraded_mode(
                    DegradationLevel.MODERATE,
                    DegradationReason.DEPENDENCY_FAILURE,
                    f"Dependency {dependency_name} is UNHEALTHY"
                )
            elif status == HealthStatus.DEGRADED:
                self.enter_degraded_mode(
                    DegradationLevel.LIGHT,
                    DegradationReason.DEPENDENCY_FAILURE,
                    f"Dependency {dependency_name} is DEGRADED"
                )
            elif (status == HealthStatus.HEALTHY and 
                  DegradationReason.DEPENDENCY_FAILURE in self.state.reasons):
                # Check if this was the only failing dependency
                # For simplicity, we're exiting the degraded mode for this reason
                # In a real implementation, you'd track which dependencies are failing
                self.exit_degraded_mode(DegradationReason.DEPENDENCY_FAILURE)
                
    def _update_active_fallbacks(self) -> None:
        """Update the set of active fallback strategies based on current state."""
        active = set()
        
        # Find applicable strategies
        for strategy in self.fallback_strategies:
            if (self.state.level >= strategy.min_degradation_level and
                    any(reason in strategy.applicable_reasons for reason in self.state.reasons)):
                active.add(strategy.operation)
                
        # Log any changes
        newly_active = active - self.active_fallbacks
        newly_inactive = self.active_fallbacks - active
        
        for op in newly_active:
            logger.info(f"Activating fallback strategy: {op.value}")
            
        for op in newly_inactive:
            logger.info(f"Deactivating fallback strategy: {op.value}")
            
        self.active_fallbacks = active
        
    def _report_status_change(self) -> None:
        """Report degradation status change via event bus."""
        if not self.event_bus:
            return
            
        try:
            # Create and publish event
            event = create_event(
                event_type=EventType.SERVICE_DEGRADED,
                source_service=self.service_name,
                data={
                    "degradation_level": self.state.level.name,
                    "reasons": [reason.value for reason in self.state.reasons],
                    "message": self.state.message,
                    "active_fallbacks": [op.value for op in self.active_fallbacks],
                    "start_time": self.state.start_time.isoformat() if self.state.start_time else None,
                    "last_update_time": self.state.last_update_time.isoformat()
                },
                priority=EventPriority.HIGH
            )
            
            self.event_bus.publish(event)
        except Exception as e:
            logger.error(f"Failed to report degradation status: {e}")
            
    def is_degraded(self) -> bool:
        """
        Check if the service is currently in a degraded mode.
        
        Returns:
            bool: True if in degraded mode, False otherwise
        """
        return self.state.level > DegradationLevel.NORMAL
        
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current degradation state.
        
        Returns:
            Dictionary with current state information
        """
        with self._lock:
            return {
                "degradation_level": self.state.level.name,
                "reasons": [reason.value for reason in self.state.reasons],
                "message": self.state.message,
                "active_fallbacks": [op.value for op in self.active_fallbacks],
                "start_time": self.state.start_time.isoformat() if self.state.start_time else None,
                "last_update_time": self.state.last_update_time.isoformat(),
                "duration_seconds": ((datetime.datetime.utcnow() - self.state.start_time).total_seconds()
                                    if self.state.start_time else 0)
            }
            
    def can_use_fallback(self, operation: FallbackOperation) -> bool:
        """
        Check if a specific fallback operation is active.
        
        Args:
            operation: The fallback operation to check
            
        Returns:
            bool: True if the fallback is active, False otherwise
        """
        return operation in self.active_fallbacks
        
    def should_process_operation(self, priority: OperationPriority) -> bool:
        """
        Check if an operation should be processed based on its priority
        and the current degradation level.
        
        Args:
            priority: The priority of the operation
            
        Returns:
            bool: True if the operation should be processed, False otherwise
        """
        with self._lock:
            # Always process critical operations
            if priority == OperationPriority.CRITICAL:
                return True
                
            # Process based on degradation level
            if self.state.level == DegradationLevel.NORMAL:
                return True  # Process all operations in normal mode
            elif self.state.level == DegradationLevel.LIGHT:
                return priority <= OperationPriority.LOW  # Everything except background
            elif self.state.level == DegradationLevel.MODERATE:
                return priority <= OperationPriority.MEDIUM  # Critical, high, and medium
            elif self.state.level == DegradationLevel.SEVERE:
                return priority <= OperationPriority.HIGH  # Only critical and high
            elif self.state.level == DegradationLevel.CRITICAL:
                return priority == OperationPriority.CRITICAL  # Only critical
                
            return False
            
    def with_fallback(
        self,
        normal_fn: Callable[..., T],
        fallback_fn: Callable[..., T],
        fallback_operation: FallbackOperation
    ) -> Callable[..., T]:
        """
        Create a function that uses a fallback implementation when appropriate.
        
        Args:
            normal_fn: Normal implementation of the function
            fallback_fn: Fallback implementation to use in degraded mode
            fallback_operation: The fallback operation type
            
        Returns:
            A function that will use the appropriate implementation
        """
        @functools.wraps(normal_fn)
        def wrapper(*args, **kwargs):
            if self.can_use_fallback(fallback_operation):
                return fallback_fn(*args, **kwargs)
            else:
                return normal_fn(*args, **kwargs)
        return wrapper


def prioritized_operation(
    priority: OperationPriority
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to mark an operation with a priority level.
    
    Args:
        priority: The priority level for this operation
        
    Returns:
        A decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if this class has a degraded_mode_manager attribute
            if hasattr(self, 'degraded_mode_manager'):
                degraded_mode_manager = self.degraded_mode_manager
                if isinstance(degraded_mode_manager, TradingGatewayDegradedMode):
                    # Check if operation should be processed
                    if not degraded_mode_manager.should_process_operation(priority):
                        operation_name = func.__name__
                        raise OperationNotAllowedException(
                            f"Operation {operation_name} with priority {priority.name} "
                            f"not allowed in degradation level "
                            f"{degraded_mode_manager.state.level.name}"
                        )
            # Execute the function normally
            return func(self, *args, **kwargs)
        
        # Store priority on the function for introspection
        wrapper._priority = priority  # type: ignore
        
        return cast(Callable[..., T], wrapper)
    
    return decorator


class OperationNotAllowedException(Exception):
    """Exception raised when an operation is not allowed due to degraded mode."""
    pass
