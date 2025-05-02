"""
Trading Gateway Degraded Mode Strategies

This module implements specialized degraded mode strategies for the Trading Gateway Service,
including order queueing, prioritization, and conservative risk controls.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import heapq
from dataclasses import dataclass, field

from core_foundations.resilience.degraded_mode import (
    DegradedModeManager, DegradedModeStrategy, with_degraded_mode, fallback_for
)
from core_foundations.exceptions.service_exceptions import DependencyUnavailableError
from core_foundations.models.trading import OrderType, OrderStatus, TradeDirection
from core_foundations.events.kafka_event_bus import KafkaEventBus
from core_foundations.events.event_schema import EventType, create_event

logger = logging.getLogger(__name__)


class OrderPriority(int, Enum):
    """Priority levels for orders in degraded mode"""
    CRITICAL = 0  # Stop loss orders (highest priority)
    HIGH = 1      # Close position orders
    MEDIUM = 2    # Reduce position orders
    LOW = 3       # New position orders (lowest priority)
    
    @classmethod
    def from_order_type(cls, order_type: OrderType, is_closing: bool = False) -> 'OrderPriority':
        """Determine priority based on order type and context"""
        if order_type == OrderType.STOP_LOSS or order_type == OrderType.STOP_MARKET:
            return cls.CRITICAL
        elif is_closing:
            return cls.HIGH
        elif order_type == OrderType.LIMIT:
            return cls.MEDIUM
        else:
            return cls.LOW


@dataclass(order=True)
class PrioritizedOrder:
    """Order with priority for the queue"""
    priority: int
    timestamp: float = field(compare=False)
    order_id: str = field(compare=False)
    order_data: Dict[str, Any] = field(compare=False)
    

class OrderQueueManager:
    """
    Manager for queueing orders when broker connectivity is lost.
    
    This class maintains a priority queue of orders and attempts to
    process them when connectivity is restored.
    
    In Phase 8, this has been enhanced to:
    - Publish status events to the event bus when entering/exiting degraded mode
    - Track degraded mode metrics for health monitoring
    - Implement advanced retry strategies with backoff
    """
    
    _instance = None
    
    def __new__(cls):
        """Create a singleton instance"""
        if cls._instance is None:
            cls._instance = super(OrderQueueManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the order queue manager"""
        if self._initialized:
            return
        
        self._queue = []  # Priority queue using heapq
        self._lock = threading.RLock()
        self._processing = False
        self._stop_event = threading.Event()
        self._order_processor = None
        self._event_bus = None
        self._is_degraded = False
        self._degraded_mode_start_time = None
        self._queue_stats = {
            "total_queued": 0,
            "total_processed": 0,
            "total_failed": 0,
            "last_success_time": None,
            "degraded_mode_incidents": 0,
            "current_queue_size": 0
        }
        self._initialized = True
    
    def set_event_bus(self, event_bus: KafkaEventBus) -> None:
        """
        Set the event bus for publishing degraded mode events.
        
        Args:
            event_bus: KafkaEventBus instance
        """
        self._event_bus = event_bus
        logger.info("Event bus connected to OrderQueueManager")
        
    def enter_degraded_mode(self, reason: str) -> None:
        """
        Enter degraded mode and publish status event.
        
        Args:
            reason: The reason for entering degraded mode
        """
        with self._lock:
            if not self._is_degraded:
                self._is_degraded = True
                self._degraded_mode_start_time = datetime.utcnow()
                self._queue_stats["degraded_mode_incidents"] += 1
                
                # Publish event if event bus is available
                if self._event_bus:
                    try:
                        event = create_event(
                            event_type=EventType.SERVICE_STATUS_CHANGED,
                            source_service="trading-gateway-service",
                            data={
                                "status": "DEGRADED",
                                "reason": reason,
                                "timestamp": self._degraded_mode_start_time.isoformat(),
                                "details": {
                                    "queue_stats": self._queue_stats
                                }
                            }
                        )
                        self._event_bus.publish(event)
                        logger.info(f"Published degraded mode event: {reason}")
                    except Exception as e:
                        logger.error(f"Failed to publish degraded mode event: {e}")
                
                logger.warning(f"Trading Gateway entered DEGRADED mode: {reason}")
    
    def exit_degraded_mode(self) -> None:
        """Exit degraded mode and publish status event."""
        with self._lock:
            if self._is_degraded:
                self._is_degraded = False
                duration = None
                
                if self._degraded_mode_start_time:
                    duration = (datetime.utcnow() - self._degraded_mode_start_time).total_seconds()
                    
                # Publish event if event bus is available
                if self._event_bus:
                    try:
                        event = create_event(
                            event_type=EventType.SERVICE_STATUS_CHANGED,
                            source_service="trading-gateway-service",
                            data={
                                "status": "HEALTHY",
                                "reason": "Connectivity restored",
                                "timestamp": datetime.utcnow().isoformat(),
                                "details": {
                                    "degraded_duration_seconds": duration,
                                    "queue_stats": self._queue_stats
                                }
                            }
                        )
                        self._event_bus.publish(event)
                        logger.info("Published service recovery event")
                    except Exception as e:
                        logger.error(f"Failed to publish service recovery event: {e}")
                
                logger.info(f"Trading Gateway exited DEGRADED mode (duration: {duration}s)")
                self._degraded_mode_start_time = None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status information for health monitoring.
        
        Returns:
            Dict with status information
        """
        with self._lock:
            return {
                "is_degraded": self._is_degraded,
                "degraded_mode_start_time": self._degraded_mode_start_time.isoformat() if self._degraded_mode_start_time else None,
                "queue_size": len(self._queue),
                "queue_stats": self._queue_stats.copy()
            }
            
    def queue_order(
        self,
        order_id: str,
        order_data: Dict[str, Any],
        priority: OrderPriority = None
    ) -> None:
        """
        Queue an order for later processing.
        
        Args:
            order_id: Unique order ID
            order_data: Order data dictionary
            priority: Order priority (or None to determine from order data)
        """
        with self._lock:
            # Auto-enter degraded mode if this is the first queued order
            if not self._is_degraded and not self._queue:
                self.enter_degraded_mode("Order queueing activated")
            
            # Determine priority if not specified
            if priority is None:
                order_type = OrderType(order_data.get('type', 'MARKET'))
                is_closing = order_data.get('is_closing', False)
                priority = OrderPriority.from_order_type(order_type, is_closing)
            
            # Create prioritized order
            prioritized_order = PrioritizedOrder(
                priority=priority.value,
                timestamp=time.time(),
                order_id=order_id,
                order_data=order_data
            )
            
            # Add to priority queue
            heapq.heappush(self._queue, prioritized_order)
            
            # Update stats
            self._queue_stats["total_queued"] += 1
            self._queue_stats["current_queue_size"] = len(self._queue)
            
            logger.info(f"Queued order {order_id} with priority {priority.name}")
    
    def set_order_processor(self, processor: Callable[[str, Dict[str, Any]], bool]) -> None:
        """
        Set the function to process orders when connectivity is restored.
        
        Args:
            processor: Function(order_id, order_data) that returns True if successful
        """
        self._order_processor = processor
    
    def start_processing(self, check_interval: float = 5.0) -> None:
        """
        Start processing queued orders.
        
        Args:
            check_interval: Interval between processing attempts in seconds
        """
        if self._processing:
            logger.warning("Order queue processor already running")
            return
        
        if self._order_processor is None:
            logger.error("Cannot start processing without an order processor")
            return
        
        self._processing = True
        self._stop_event.clear()
        
        def processing_loop():
            while not self._stop_event.is_set():
                try:
                    # Process any queued orders
                    self._process_queue()
                except Exception as e:
                    logger.error(f"Error in order queue processing loop: {e}")
                
                # Wait for next check
                self._stop_event.wait(check_interval)
        
        # Start processing thread
        thread = threading.Thread(
            target=processing_loop,
            daemon=True,
            name="order-queue-processor"
        )
        thread.start()
        
        logger.info(f"Started order queue processor (interval: {check_interval}s)")
    
    def stop_processing(self) -> None:
        """Stop processing queued orders"""
        self._stop_event.set()
        self._processing = False
        logger.info("Stopped order queue processor")

    def _process_queue(self) -> int:
        \"\"\"
        Process queued orders.
        
        Returns:
            Number of orders processed
        \"\"\"
        if self._order_processor is None:
            return 0
        
        processed_count = 0
        failed_orders = []
        
        with self._lock:
            # Process orders in priority order
            initial_queue_size = len(self._queue)
            
            # Don't try to process if empty
            if not self._queue:
                # If the queue is empty and we were in degraded mode, exit it.
                if self._is_degraded:
                    self.exit_degraded_mode()
                return 0
                
            # Update queue size stats
            self._queue_stats["current_queue_size"] = initial_queue_size
            
            while self._queue:
                order = heapq.heappop(self._queue)
                
                try:
                    # Try to process order
                    success = self._order_processor(order.order_id, order.order_data)
                    
                    if success:
                        processed_count += 1
                        self._queue_stats["total_processed"] += 1
                        self._queue_stats["last_success_time"] = datetime.utcnow().isoformat()
                        logger.info(f"Processed queued order {order.order_id}")
                        
                        # Notify about processed order via event bus if available
                        if self._event_bus:
                            try:
                                event = create_event(
                                    event_type=EventType.ORDER_PROCESSED_FROM_QUEUE,
                                    source_service="trading-gateway-service",
                                    data={
                                        "order_id": order.order_id,
                                        "queue_delay_seconds": time.time() - order.timestamp,
                                        "priority": OrderPriority(order.priority).name
                                    }
                                )
                                self._event_bus.publish(event)
                            except Exception as e:
                                logger.error(f"Failed to publish order processed event: {e}")
                    else:
                        # Put back in queue if processing failed
                        failed_orders.append(order)
                        self._queue_stats["total_failed"] += 1
                        logger.warning(f"Failed to process queued order {order.order_id}")
                except Exception as e:
                    # Handle unexpected errors
                    failed_orders.append(order)
                    self._queue_stats["total_failed"] += 1
                    logger.error(f"Error processing queued order {order.order_id}: {e}")
                    
            # Add failed orders back to queue
            for order in failed_orders:
                heapq.heappush(self._queue, order)
            
            # Update queue size stats again
            self._queue_stats["current_queue_size"] = len(self._queue)
            
            # If we've processed everything successfully, exit degraded mode
            if processed_count > 0 and not self._queue:
                self.exit_degraded_mode()
                
            return processed_count
    
    def get_queue_size(self) -> int:
        \"\"\"Get number of orders in queue\"\"\"
        with self._lock:
            return len(self._queue)
    
    def get_queue_summary(self) -> Dict[str, Any]:
        \"\"\"Get summary of queued orders\"\"\"
        with self._lock:
            priorities = {}
            oldest = None
            newest = None
            
            for order in self._queue:
                priority = OrderPriority(order.priority).name
                priorities[priority] = priorities.get(priority, 0) + 1
                
                if oldest is None or order.timestamp < oldest:
                    oldest = order.timestamp
                
                if newest is None or order.timestamp > newest:
                    newest = order.timestamp
            
            return {
                "total": len(self._queue),
                "priorities": priorities,
                "oldest": datetime.fromtimestamp(oldest).isoformat() if oldest else None,
                "newest": datetime.fromtimestamp(newest).isoformat() if newest else None,
            }
    
    def clear_queue(self) -> int:
        \"\"\"
        Clear the order queue.
        
        Returns:
            Number of orders cleared
        \"\"\"
        with self._lock:
            count = len(self._queue)
            self._queue = []
            logger.info(f"Cleared order queue ({count} orders)")
            return count


class RiskControlManager:
    """
    Manager for applying conservative risk controls when risk management service is unavailable.
    
    This class implements fallback risk controls that are more conservative
    than the normal risk management service.
    """
    
    _instance = None
    
    def __new__(cls):
        """Create a singleton instance"""
        if cls._instance is None:
            cls._instance = super(RiskControlManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the risk control manager"""
        if self._initialized:
            return
        
        # Conservative risk limits
        self._max_position_size = {}  # symbol -> max size
        self._max_order_size = {}     # symbol -> max size
        self._max_daily_loss = {}     # account -> max loss
        self._max_open_positions = {}  # account -> max positions
        self._position_sizes = {}     # account -> {symbol -> size}
        
        # Current state tracking
        self._daily_pnl = {}          # account -> current P&L
        
        self._lock = threading.RLock()
        self._initialized = True
    
    def set_conservative_limits(
        self,
        max_position_size: Dict[str, float] = None,
        max_order_size: Dict[str, float] = None,
        max_daily_loss: Dict[str, float] = None,
        max_open_positions: Dict[str, int] = None
    ) -> None:
        """
        Set conservative risk limits for degraded mode.
        
        Args:
            max_position_size: Maximum position size by symbol
            max_order_size: Maximum order size by symbol
            max_daily_loss: Maximum daily loss by account
            max_open_positions: Maximum open positions by account
        """
        with self._lock:
            if max_position_size:
                self._max_position_size = max_position_size
            
            if max_order_size:
                self._max_order_size = max_order_size
            
            if max_daily_loss:
                self._max_daily_loss = max_daily_loss
            
            if max_open_positions:
                self._max_open_positions = max_open_positions
    
    def set_default_limits(self, account_id: str, account_balance: float) -> None:
        """
        Set default conservative limits based on account balance.
        
        Args:
            account_id: Account ID
            account_balance: Current account balance
        """
        with self._lock:
            # Default to 1% of account balance per position
            max_position = account_balance * 0.01
            
            # Default to 0.5% of account balance per order
            max_order = account_balance * 0.005
            
            # Default to 5% max daily loss
            max_loss = account_balance * 0.05
            
            # Default to 5 open positions
            max_positions = 5
            
            # Apply defaults
            self._max_daily_loss[account_id] = max_loss
            self._max_open_positions[account_id] = max_positions
            
            # Set for common forex pairs
            common_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"]
            for pair in common_pairs:
                self._max_position_size[pair] = max_position
                self._max_order_size[pair] = max_order
    
    def update_position(self, account_id: str, symbol: str, size: float) -> None:
        """
        Update current position size for tracking.
        
        Args:
            account_id: Account ID
            symbol: Currency pair symbol
            size: Position size (positive for long, negative for short)
        """
        with self._lock:
            if account_id not in self._position_sizes:
                self._position_sizes[account_id] = {}
            
            # Update position size
            self._position_sizes[account_id][symbol] = size
    
    def update_pnl(self, account_id: str, daily_pnl: float) -> None:
        """
        Update current daily P&L for tracking.
        
        Args:
            account_id: Account ID
            daily_pnl: Current daily P&L
        """
        with self._lock:
            self._daily_pnl[account_id] = daily_pnl
    
    def check_order(
        self,
        account_id: str,
        symbol: str,
        direction: str,
        size: float,
        is_closing: bool = False
    ) -> Tuple[bool, str]:
        """
        Check if an order meets conservative risk controls.
        
        Args:
            account_id: Account ID
            symbol: Currency pair symbol
            direction: Trade direction (BUY or SELL)
            size: Order size
            is_closing: Whether this order closes an existing position
            
        Returns:
            Tuple of (allowed, reason)
        """
        with self._lock:
            # Always allow closing orders
            if is_closing:
                return True, ""
            
            # Check order size limit
            max_order = self._max_order_size.get(symbol, 0.01)
            if size > max_order:
                return False, f"Order size {size} exceeds degraded mode limit {max_order} for {symbol}"
            
            # Check daily loss limit
            daily_pnl = self._daily_pnl.get(account_id, 0)
            max_loss = self._max_daily_loss.get(account_id, -1000)
            if daily_pnl < max_loss:
                return False, f"Daily loss {daily_pnl} exceeds degraded mode limit {max_loss} for account {account_id}"
            
            # Check maximum open positions
            current_positions = len(self._position_sizes.get(account_id, {}))
            max_positions = self._max_open_positions.get(account_id, 5)
            if current_positions >= max_positions:
                return False, f"Open positions ({current_positions}) at degraded mode limit {max_positions}"
            
            # Check position size limit after this order
            current_size = self._position_sizes.get(account_id, {}).get(symbol, 0)
            new_size = current_size
            
            if direction == "BUY":
                new_size += size
            else:
                new_size -= size
            
            max_position = self._max_position_size.get(symbol, 0.02)
            if abs(new_size) > max_position:
                return False, f"Position size {abs(new_size)} would exceed degraded mode limit {max_position} for {symbol}"
            
            # All checks passed
            return True, ""
    
    def get_risk_status(self, account_id: str) -> Dict[str, Any]:
        """Get current risk status"""
        with self._lock:
            return {
                "daily_pnl": self._daily_pnl.get(account_id, 0),
                "max_daily_loss": self._max_daily_loss.get(account_id, 0),
                "open_positions": len(self._position_sizes.get(account_id, {})),
                "max_open_positions": self._max_open_positions.get(account_id, 0),
                "position_sizes": self._position_sizes.get(account_id, {}),
                "degraded_mode_active": True
            }


# Decorate main trading gateway functions with degraded mode handling

def submit_order_degraded_broker(
    original_submit_order_func: Callable,
    order_id: str,
    order_data: Dict[str, Any]
):
    """
    Fallback for broker connectivity issues.
    
    This fallback queues orders when the broker is unavailable.
    
    Args:
        original_submit_order_func: Original submit order function
        order_id: Order ID
        order_data: Order data
        
    Returns:
        Order ID and status
    """
    # Get order queue manager
    queue_manager = OrderQueueManager()
    
    # Queue the order
    queue_manager.queue_order(order_id, order_data)
    
    # Return order as PENDING
    return {
        "order_id": order_id,
        "status": OrderStatus.PENDING.value,
        "message": "Order queued due to broker connectivity issues",
        "queue_position": queue_manager.get_queue_size()
    }


def check_risk_degraded(
    original_check_risk_func: Callable,
    account_id: str,
    symbol: str,
    direction: str,
    size: float,
    is_closing: bool = False
):
    """
    Fallback for risk management service.
    
    This fallback applies conservative risk controls when the risk management service is unavailable.
    
    Args:
        original_check_risk_func: Original risk check function
        account_id: Account ID
        symbol: Currency pair symbol
        direction: Trade direction
        size: Order size
        is_closing: Whether this order closes an existing position
        
    Returns:
        Tuple of (allowed, reason)
    """
    # Get risk control manager
    risk_manager = RiskControlManager()
    
    # Apply conservative risk check
    return risk_manager.check_order(account_id, symbol, direction, size, is_closing)


# Helper function to set up degraded mode for trading gateway
def configure_trading_gateway_degraded_mode(
    submit_order_func: Callable,
    check_risk_func: Callable,
    process_queue_func: Callable,
    account_balance_func: Callable
):
    """
    Configure degraded mode for trading gateway.
    
    Args:
        submit_order_func: Function to submit orders to broker
        check_risk_func: Function to check risk limits
        process_queue_func: Function to process queued orders
        account_balance_func: Function to get account balance
    """
    # Set up fallbacks
    fallback_for("broker", submit_order_func)(
        lambda *args, **kwargs: submit_order_degraded_broker(submit_order_func, *args, **kwargs)
    )
    
    fallback_for("risk-management", check_risk_func)(
        lambda *args, **kwargs: check_risk_degraded(check_risk_func, *args, **kwargs)
    )
    
    # Configure order queue processor
    queue_manager = OrderQueueManager()
    queue_manager.set_order_processor(process_queue_func)
    queue_manager.start_processing()
    
    # Configure risk manager with default limits
    risk_manager = RiskControlManager()
    
    # Set up dependency status handlers
    degraded_manager = DegradedModeManager()
    
    def handle_degraded_status(dependency_name: str, is_degraded: bool):
        """Handle changes in dependency status"""
        if dependency_name == "broker":
            if is_degraded:
                logger.warning("Broker connectivity lost, enabling order queue")
                queue_manager.start_processing()
            else:
                logger.info("Broker connectivity restored")
        
        elif dependency_name == "risk-management":
            if is_degraded:
                logger.warning("Risk management service unavailable, using conservative limits")
                
                # Set default limits based on account balance
                try:
                    for account_id, balance in account_balance_func().items():
                        risk_manager.set_default_limits(account_id, balance)
                except Exception as e:
                    logger.error(f"Error setting default risk limits: {e}")
                    # Set some minimal defaults
                    risk_manager.set_default_limits("default", 10000)
            else:
                logger.info("Risk management service restored")
    
    # Register handler
    degraded_manager.add_degraded_mode_handler(handle_degraded_status)
    
    logger.info("Configured trading gateway degraded mode strategies")
