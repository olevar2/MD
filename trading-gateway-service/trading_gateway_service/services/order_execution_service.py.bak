"""
Order Execution Service for the trading gateway.

This service is responsible for processing order requests, routing them to appropriate
brokers, and managing the order lifecycle. It also provides advanced execution algorithms
for optimizing order execution.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
import uuid
from enum import Enum

from ..interfaces.broker_adapter_interface import (
    BrokerAdapterInterface,
    OrderRequest,
    ExecutionReport,
    OrderStatus,
)
from ..execution_algorithms import (
    BaseExecutionAlgorithm,
    SmartOrderRoutingAlgorithm,
    TWAPAlgorithm,
    VWAPAlgorithm,
    ImplementationShortfallAlgorithm,
)


class ExecutionMode(Enum):
    """Execution modes for the order execution service."""
    LIVE = "live"  # Real trading with actual broker
    PAPER = "paper"  # Paper trading with simulated fills
    SIMULATED = "simulated"  # Full simulation with no broker connection
    BACKTEST = "backtest"  # Historical backtest mode


class ExecutionAlgorithm(Enum):
    """Available execution algorithms."""
    DIRECT = "direct"  # Direct execution with a single broker
    SOR = "sor"  # Smart Order Routing across multiple brokers
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"  # Implementation Shortfall


class OrderExecutionService:
    """
    Service for executing orders through broker adapters.
    Handles order routing, execution, and state management.
    """

    def __init__(self, mode: ExecutionMode = ExecutionMode.SIMULATED):
        """
        Initialize the order execution service.

        Args:
            mode: The execution mode to operate in
        """
        self.logger = logging.getLogger(__name__)
        self.mode = mode
        self.broker_adapters: Dict[str, BrokerAdapterInterface] = {}
        self.default_broker = None
        self.orders: Dict[str, Dict[str, Any]] = {}  # Track all orders by ID
        self.active_algorithms: Dict[str, BaseExecutionAlgorithm] = {}  # Track active execution algorithms
        self.callbacks: Dict[str, List[Callable]] = {
            "order_placed": [],
            "order_filled": [],
            "order_cancelled": [],
            "order_rejected": [],
            "order_modified": [],
            "execution_error": [],
            "algorithm_started": [],
            "algorithm_progress": [],
            "algorithm_completed": [],
            "algorithm_failed": [],
        }

        # Default algorithm configurations
        self.algorithm_configs: Dict[str, Dict[str, Any]] = {
            ExecutionAlgorithm.SOR.value: {},
            ExecutionAlgorithm.TWAP.value: {
                "duration_minutes": 60,
                "num_slices": 12,
            },
            ExecutionAlgorithm.VWAP.value: {
                "duration_minutes": 60,
                "num_slices": 12,
            },
            ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL.value: {
                "max_duration_minutes": 120,
                "urgency": 0.5,
            },
        }

        self.logger.info(f"OrderExecutionService initialized in {mode.value} mode")

    def register_broker_adapter(self, name: str, adapter: BrokerAdapterInterface, default: bool = False) -> bool:
        """
        Register a broker adapter with the execution service.

        Args:
            name: Name identifier for the broker adapter
            adapter: The broker adapter instance
            default: Whether this should be the default broker

        Returns:
            True if registration was successful
        """
        if name in self.broker_adapters:
            self.logger.warning(f"Broker adapter '{name}' already registered. Overwriting.")

        self.broker_adapters[name] = adapter
        if default or self.default_broker is None:
            self.default_broker = name
            self.logger.info(f"Set '{name}' as default broker")

        self.logger.info(f"Registered broker adapter: {name}")
        return True

    def connect(self, broker_name: Optional[str] = None, credentials: Optional[Dict[str, str]] = None) -> bool:
        """
        Connect to a broker or all registered brokers.

        Args:
            broker_name: Name of specific broker to connect to, or None for all
            credentials: Authentication credentials (if needed)

        Returns:
            True if connection successful, False if any connection failed
        """
        if self.mode == ExecutionMode.BACKTEST:
            self.logger.info("In backtest mode, no broker connection needed.")
            return True

        success = True

        if broker_name:
            if broker_name not in self.broker_adapters:
                self.logger.error(f"Broker '{broker_name}' not registered")
                return False

            adapters_to_connect = {broker_name: self.broker_adapters[broker_name]}
        else:
            adapters_to_connect = self.broker_adapters

        for name, adapter in adapters_to_connect.items():
            try:
                broker_credentials = credentials.get(name, {}) if credentials else {}
                if not adapter.connect(broker_credentials):
                    self.logger.error(f"Failed to connect to broker: {name}")
                    success = False
                else:
                    self.logger.info(f"Connected to broker: {name}")
            except Exception as e:
                self.logger.error(f"Error connecting to broker {name}: {str(e)}")
                success = False

        return success

    def disconnect(self, broker_name: Optional[str] = None) -> bool:
        """
        Disconnect from a broker or all registered brokers.

        Args:
            broker_name: Name of specific broker to disconnect from, or None for all

        Returns:
            True if disconnection successful
        """
        if broker_name:
            if broker_name not in self.broker_adapters:
                self.logger.error(f"Broker '{broker_name}' not registered")
                return False

            adapters_to_disconnect = {broker_name: self.broker_adapters[broker_name]}
        else:
            adapters_to_disconnect = self.broker_adapters

        success = True
        for name, adapter in adapters_to_disconnect.items():
            try:
                if not adapter.disconnect():
                    self.logger.error(f"Failed to disconnect from broker: {name}")
                    success = False
                else:
                    self.logger.info(f"Disconnected from broker: {name}")
            except Exception as e:
                self.logger.error(f"Error disconnecting from broker {name}: {str(e)}")
                success = False

        return success

    def place_order(self, order: OrderRequest, broker_name: Optional[str] = None,
                 algorithm: Optional[Union[str, ExecutionAlgorithm]] = None,
                 algorithm_config: Optional[Dict[str, Any]] = None) -> ExecutionReport:
        """
        Place an order with a specified broker or using an execution algorithm.

        Args:
            order: Order request to be placed
            broker_name: Name of the broker to use, or None for default
            algorithm: Execution algorithm to use (None for direct execution)
            algorithm_config: Configuration for the execution algorithm

        Returns:
            Execution report of the order placement
        """
        # Convert string algorithm to enum if needed
        if isinstance(algorithm, str):
            try:
                algorithm = ExecutionAlgorithm(algorithm.lower())
            except ValueError:
                self.logger.warning(f"Unknown algorithm: {algorithm}, using direct execution")
                algorithm = None

        # If using an algorithm other than direct execution, delegate to the algorithm
        if algorithm and algorithm != ExecutionAlgorithm.DIRECT:
            return asyncio.run(self._execute_with_algorithm(order, algorithm, algorithm_config))

        # Direct execution with a single broker
        if broker_name is None:
            broker_name = self.default_broker

        if broker_name not in self.broker_adapters:
            error_msg = f"Broker '{broker_name}' not registered"
            self.logger.error(error_msg)
            return ExecutionReport(
                order_id=str(uuid.uuid4()),
                client_order_id=order.client_order_id,
                instrument=order.instrument,
                status=OrderStatus.REJECTED,
                direction=order.direction,
                order_type=order.order_type,
                quantity=order.quantity,
                filled_quantity=0.0,
                price=order.price,
                rejection_reason=error_msg,
            )

        # Handle different execution modes
        if self.mode == ExecutionMode.BACKTEST:
            self.logger.warning("Placing order in backtest mode - this should be handled by the backtester")
            # Return dummy execution report for backtesting
            return self._create_dummy_execution_report(order, OrderStatus.FILLED)

        elif self.mode == ExecutionMode.SIMULATED:
            # In simulation mode, we create a simulated execution report
            # This would normally be handled by a proper simulator component
            self.logger.info(f"Simulating order placement: {order.instrument} {order.direction.value} {order.quantity}")
            execution_report = self._create_dummy_execution_report(order, OrderStatus.PENDING)

        elif self.mode == ExecutionMode.PAPER:
            # Paper trading should use real market data but simulated execution
            # This would be handled by a paper trading component
            self.logger.info(f"Paper trading order: {order.instrument} {order.direction.value} {order.quantity}")
            execution_report = self._create_dummy_execution_report(order, OrderStatus.PENDING)

        else:  # LIVE mode
            # Forward to the actual broker adapter
            adapter = self.broker_adapters[broker_name]
            try:
                if not adapter.is_connected():
                    self.logger.warning(f"Broker '{broker_name}' not connected. Attempting to connect...")
                    adapter.connect({})  # Attempt reconnection with empty credentials

                self.logger.info(f"Placing order with broker {broker_name}: {order.instrument} {order.direction.value} {order.quantity}")
                execution_report = adapter.place_order(order)
            except Exception as e:
                error_msg = f"Error placing order with broker {broker_name}: {str(e)}"
                self.logger.error(error_msg)
                execution_report = ExecutionReport(
                    order_id=str(uuid.uuid4()),
                    client_order_id=order.client_order_id,
                    instrument=order.instrument,
                    status=OrderStatus.REJECTED,
                    direction=order.direction,
                    order_type=order.order_type,
                    quantity=order.quantity,
                    filled_quantity=0.0,
                    price=order.price,
                    rejection_reason=error_msg,
                )
                self._trigger_callbacks("execution_error", execution_report)

        # Store the order for tracking
        if execution_report.status != OrderStatus.REJECTED:
            self.orders[execution_report.order_id] = {
                "order": order,
                "execution_report": execution_report,
                "broker": broker_name,
                "updates": [],
                "created_at": datetime.utcnow(),
            }
            self._trigger_callbacks("order_placed", execution_report)

        return execution_report

    def cancel_order(self, order_id: str) -> ExecutionReport:
        """
        Cancel an existing order.

        Args:
            order_id: ID of the order to cancel

        Returns:
            Execution report of the cancellation
        """
        if order_id not in self.orders:
            error_msg = f"Order ID {order_id} not found"
            self.logger.error(error_msg)
            return ExecutionReport(
                order_id=order_id,
                client_order_id="unknown",
                instrument="unknown",
                status=OrderStatus.REJECTED,
                direction=None,  # Type will be corrected by ExecutionReport.__post_init__
                order_type=None,  # Type will be corrected by ExecutionReport.__post_init__
                quantity=0.0,
                filled_quantity=0.0,
                price=None,
                rejection_reason=error_msg,
            )

        order_info = self.orders[order_id]
        broker_name = order_info["broker"]

        if self.mode == ExecutionMode.BACKTEST:
            self.logger.warning("Cancelling order in backtest mode - this should be handled by the backtester")
            return self._create_dummy_execution_report(order_info["order"], OrderStatus.CANCELLED)

        elif self.mode in [ExecutionMode.SIMULATED, ExecutionMode.PAPER]:
            # Simulate cancellation
            execution_report = ExecutionReport(
                order_id=order_id,
                client_order_id=order_info["execution_report"].client_order_id,
                instrument=order_info["execution_report"].instrument,
                status=OrderStatus.CANCELLED,
                direction=order_info["execution_report"].direction,
                order_type=order_info["execution_report"].order_type,
                quantity=order_info["execution_report"].quantity,
                filled_quantity=order_info["execution_report"].filled_quantity,
                price=order_info["execution_report"].price,
            )
        else:  # LIVE mode
            adapter = self.broker_adapters[broker_name]
            try:
                self.logger.info(f"Cancelling order {order_id} with broker {broker_name}")
                execution_report = adapter.cancel_order(order_id)
            except Exception as e:
                error_msg = f"Error cancelling order with broker {broker_name}: {str(e)}"
                self.logger.error(error_msg)
                execution_report = ExecutionReport(
                    order_id=order_id,
                    client_order_id=order_info["execution_report"].client_order_id,
                    instrument=order_info["execution_report"].instrument,
                    status=OrderStatus.REJECTED,
                    direction=order_info["execution_report"].direction,
                    order_type=order_info["execution_report"].order_type,
                    quantity=order_info["execution_report"].quantity,
                    filled_quantity=order_info["execution_report"].filled_quantity,
                    price=order_info["execution_report"].price,
                    rejection_reason=error_msg,
                )
                self._trigger_callbacks("execution_error", execution_report)

        # Update order info
        if execution_report.status == OrderStatus.CANCELLED:
            order_info["execution_report"] = execution_report
            order_info["updates"].append({
                "timestamp": datetime.utcnow(),
                "status": execution_report.status,
                "filled_quantity": execution_report.filled_quantity,
            })
            self._trigger_callbacks("order_cancelled", execution_report)

        return execution_report

    def modify_order(self, order_id: str, modifications: Dict[str, Any]) -> ExecutionReport:
        """
        Modify an existing order.

        Args:
            order_id: ID of the order to modify
            modifications: Dictionary of parameters to modify

        Returns:
            Execution report of the modification
        """
        if order_id not in self.orders:
            error_msg = f"Order ID {order_id} not found"
            self.logger.error(error_msg)
            return ExecutionReport(
                order_id=order_id,
                client_order_id="unknown",
                instrument="unknown",
                status=OrderStatus.REJECTED,
                direction=None,  # Type will be corrected by ExecutionReport.__post_init__
                order_type=None,  # Type will be corrected by ExecutionReport.__post_init__
                quantity=0.0,
                filled_quantity=0.0,
                price=None,
                rejection_reason=error_msg,
            )

        order_info = self.orders[order_id]
        broker_name = order_info["broker"]

        if self.mode == ExecutionMode.BACKTEST:
            self.logger.warning("Modifying order in backtest mode - this should be handled by the backtester")
            # Create dummy execution report with modifications applied
            modified_order = order_info["order"]
            # Apply modifications to a copy of the order
            for key, value in modifications.items():
                if hasattr(modified_order, key):
                    setattr(modified_order, key, value)
            return self._create_dummy_execution_report(modified_order, OrderStatus.OPEN)

        elif self.mode in [ExecutionMode.SIMULATED, ExecutionMode.PAPER]:
            # Simulate modification
            prev_report = order_info["execution_report"]
            execution_report = ExecutionReport(
                order_id=order_id,
                client_order_id=prev_report.client_order_id,
                instrument=prev_report.instrument,
                status=prev_report.status,
                direction=prev_report.direction,
                order_type=prev_report.order_type,
                quantity=modifications.get("quantity", prev_report.quantity),
                filled_quantity=prev_report.filled_quantity,
                price=modifications.get("price", prev_report.price),
                stop_loss=modifications.get("stop_loss", prev_report.stop_loss),
                take_profit=modifications.get("take_profit", prev_report.take_profit),
            )
        else:  # LIVE mode
            adapter = self.broker_adapters[broker_name]
            try:
                self.logger.info(f"Modifying order {order_id} with broker {broker_name}: {modifications}")
                execution_report = adapter.modify_order(order_id, modifications)
            except Exception as e:
                error_msg = f"Error modifying order with broker {broker_name}: {str(e)}"
                self.logger.error(error_msg)
                execution_report = ExecutionReport(
                    order_id=order_id,
                    client_order_id=order_info["execution_report"].client_order_id,
                    instrument=order_info["execution_report"].instrument,
                    status=OrderStatus.REJECTED,
                    direction=order_info["execution_report"].direction,
                    order_type=order_info["execution_report"].order_type,
                    quantity=order_info["execution_report"].quantity,
                    filled_quantity=order_info["execution_report"].filled_quantity,
                    price=order_info["execution_report"].price,
                    rejection_reason=error_msg,
                )
                self._trigger_callbacks("execution_error", execution_report)

        # Update order info
        if execution_report.status != OrderStatus.REJECTED:
            order_info["execution_report"] = execution_report
            order_info["updates"].append({
                "timestamp": datetime.utcnow(),
                "status": execution_report.status,
                "modifications": modifications,
            })
            self._trigger_callbacks("order_modified", execution_report)

        return execution_report

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details about a specific order.

        Args:
            order_id: ID of the order to retrieve

        Returns:
            Dictionary with order details or None if not found
        """
        if order_id not in self.orders:
            self.logger.warning(f"Order ID {order_id} not found")
            return None

        return self.orders[order_id]

    def get_orders(self, instrument: Optional[str] = None, status: Optional[OrderStatus] = None) -> List[Dict[str, Any]]:
        """
        Get orders matching specified criteria.

        Args:
            instrument: Filter orders by instrument
            status: Filter orders by status

        Returns:
            List of order dictionaries
        """
        filtered_orders = []

        for order_info in self.orders.values():
            execution_report = order_info["execution_report"]

            if instrument and execution_report.instrument != instrument:
                continue

            if status and execution_report.status != status:
                continue

            filtered_orders.append(order_info)

        return filtered_orders

    def update_execution_status(self, order_id: str, status_update: Dict[str, Any]) -> bool:
        """
        Update the status of an order (e.g., from external callbacks/websockets).

        Args:
            order_id: ID of the order to update
            status_update: New status information

        Returns:
            True if update was successful
        """
        if order_id not in self.orders:
            self.logger.warning(f"Cannot update status for unknown order ID: {order_id}")
            return False

        order_info = self.orders[order_id]
        prev_report = order_info["execution_report"]

        # Create updated execution report
        new_status = status_update.get("status", prev_report.status)
        filled_qty = status_update.get("filled_quantity", prev_report.filled_quantity)
        executed_price = status_update.get("executed_price", prev_report.executed_price)

        updated_report = ExecutionReport(
            order_id=order_id,
            client_order_id=prev_report.client_order_id,
            instrument=prev_report.instrument,
            status=new_status,
            direction=prev_report.direction,
            order_type=prev_report.order_type,
            quantity=prev_report.quantity,
            filled_quantity=filled_qty,
            price=prev_report.price,
            executed_price=executed_price,
            stop_loss=prev_report.stop_loss,
            take_profit=prev_report.take_profit,
        )

        # Update order info
        order_info["execution_report"] = updated_report
        order_info["updates"].append({
            "timestamp": datetime.utcnow(),
            "status": new_status,
            "filled_quantity": filled_qty,
            "executed_price": executed_price,
        })

        # Trigger appropriate callbacks
        if new_status == OrderStatus.FILLED and prev_report.status != OrderStatus.FILLED:
            self._trigger_callbacks("order_filled", updated_report)
        elif new_status == OrderStatus.PARTIALLY_FILLED:
            self._trigger_callbacks("order_partially_filled", updated_report)
        elif new_status == OrderStatus.REJECTED and prev_report.status != OrderStatus.REJECTED:
            self._trigger_callbacks("order_rejected", updated_report)
        elif new_status == OrderStatus.CANCELLED and prev_report.status != OrderStatus.CANCELLED:
            self._trigger_callbacks("order_cancelled", updated_report)

        return True

    def register_callback(self, event_type: str, callback: Callable) -> bool:
        """
        Register a callback for order events.

        Args:
            event_type: Type of event to register for (order_placed, order_filled, etc.)
            callback: Function to call when event occurs

        Returns:
            True if registration was successful
        """
        if event_type not in self.callbacks:
            self.logger.error(f"Unknown event type: {event_type}")
            return False

        self.callbacks[event_type].append(callback)
        self.logger.debug(f"Registered callback for event: {event_type}")
        return True

    def _trigger_callbacks(self, event_type: str, data: Any) -> None:
        """
        Trigger all callbacks registered for an event type.

        Args:
            event_type: Type of event that occurred
            data: Data to pass to callbacks
        """
        if event_type not in self.callbacks:
            self.logger.error(f"Unknown event type: {event_type}")
            return

        for callback in self.callbacks[event_type]:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error in callback for {event_type}: {str(e)}")

    def _create_dummy_execution_report(self, order: OrderRequest, status: OrderStatus) -> ExecutionReport:
        """
        Create a dummy execution report for simulation/backtest modes.

        Args:
            order: Original order request
            status: Status to assign to the execution report

        Returns:
            Execution report with dummy values
        """
        return ExecutionReport(
            order_id=str(uuid.uuid4()),
            client_order_id=order.client_order_id,
            instrument=order.instrument,
            status=status,
            direction=order.direction,
            order_type=order.order_type,
            quantity=order.quantity,
            filled_quantity=order.quantity if status == OrderStatus.FILLED else 0.0,
            price=order.price,
            executed_price=order.price if status == OrderStatus.FILLED else None,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
        )

    async def _execute_with_algorithm(self,
                                    order: OrderRequest,
                                    algorithm: ExecutionAlgorithm,
                                    algorithm_config: Optional[Dict[str, Any]] = None) -> ExecutionReport:
        """
        Execute an order using the specified algorithm.

        Args:
            order: Order to execute
            algorithm: Algorithm to use
            algorithm_config: Configuration for the algorithm

        Returns:
            Execution report for the order
        """
        # Merge default config with provided config
        config = self.algorithm_configs.get(algorithm.value, {}).copy()
        if algorithm_config:
            config.update(algorithm_config)

        # Create the appropriate algorithm instance
        algo_instance = self._create_algorithm_instance(algorithm, config)

        if not algo_instance:
            self.logger.error(f"Failed to create algorithm instance for {algorithm.value}")
            return ExecutionReport(
                order_id=str(uuid.uuid4()),
                client_order_id=order.client_order_id,
                instrument=order.instrument,
                status=OrderStatus.REJECTED,
                direction=order.direction,
                order_type=order.order_type,
                quantity=order.quantity,
                filled_quantity=0.0,
                price=order.price,
                rejection_reason=f"Failed to create algorithm instance for {algorithm.value}",
            )

        # Register callbacks
        algo_instance.register_callback('started', lambda data: self._trigger_callbacks('algorithm_started', data))
        algo_instance.register_callback('progress', lambda data: self._trigger_callbacks('algorithm_progress', data))
        algo_instance.register_callback('completed', lambda data: self._trigger_callbacks('algorithm_completed', data))
        algo_instance.register_callback('failed', lambda data: self._trigger_callbacks('algorithm_failed', data))

        # Store the algorithm instance
        self.active_algorithms[algo_instance.algorithm_id] = algo_instance

        try:
            # Execute the algorithm
            self.logger.info(f"Executing order with {algorithm.value} algorithm: {order.instrument} {order.direction.value} {order.quantity}")
            result = await algo_instance.execute(order)

            # Create an execution report from the result
            if result.status == 'COMPLETED':
                status = OrderStatus.FILLED
            elif result.status == 'PARTIAL':
                status = OrderStatus.PARTIALLY_FILLED
            elif result.status == 'CANCELLED':
                status = OrderStatus.CANCELLED
            else:
                status = OrderStatus.REJECTED

            # Create the execution report
            execution_report = ExecutionReport(
                order_id=result.algorithm_id,
                client_order_id=order.client_order_id,
                instrument=order.instrument,
                status=status,
                direction=order.direction,
                order_type=order.order_type,
                quantity=order.quantity,
                filled_quantity=result.total_filled_quantity,
                price=order.price,
                executed_price=result.average_execution_price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
            )

            # Store the order for tracking
            if status != OrderStatus.REJECTED:
                self.orders[execution_report.order_id] = {
                    "order": order,
                    "execution_report": execution_report,
                    "broker": "algorithm",
                    "algorithm": algorithm.value,
                    "algorithm_id": result.algorithm_id,
                    "updates": [],
                    "created_at": datetime.utcnow(),
                    "metrics": result.metrics
                }
                self._trigger_callbacks("order_placed", execution_report)

            return execution_report

        except Exception as e:
            self.logger.error(f"Error executing algorithm {algorithm.value}: {str(e)}")
            return ExecutionReport(
                order_id=str(uuid.uuid4()),
                client_order_id=order.client_order_id,
                instrument=order.instrument,
                status=OrderStatus.REJECTED,
                direction=order.direction,
                order_type=order.order_type,
                quantity=order.quantity,
                filled_quantity=0.0,
                price=order.price,
                rejection_reason=f"Algorithm execution error: {str(e)}",
            )
        finally:
            # Remove the algorithm instance
            if algo_instance.algorithm_id in self.active_algorithms:
                del self.active_algorithms[algo_instance.algorithm_id]

    def _create_algorithm_instance(self,
                                 algorithm: ExecutionAlgorithm,
                                 config: Dict[str, Any]) -> Optional[BaseExecutionAlgorithm]:
        """
        Create an instance of the specified algorithm.

        Args:
            algorithm: Algorithm to create
            config: Configuration for the algorithm

        Returns:
            Algorithm instance, or None if creation failed
        """
        try:
            if algorithm == ExecutionAlgorithm.SOR:
                return SmartOrderRoutingAlgorithm(
                    broker_adapters=self.broker_adapters,
                    logger=self.logger,
                    config=config
                )
            elif algorithm == ExecutionAlgorithm.TWAP:
                return TWAPAlgorithm(
                    broker_adapters=self.broker_adapters,
                    logger=self.logger,
                    config=config
                )
            elif algorithm == ExecutionAlgorithm.VWAP:
                return VWAPAlgorithm(
                    broker_adapters=self.broker_adapters,
                    logger=self.logger,
                    config=config
                )
            elif algorithm == ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL:
                return ImplementationShortfallAlgorithm(
                    broker_adapters=self.broker_adapters,
                    logger=self.logger,
                    config=config
                )
            else:
                self.logger.error(f"Unknown algorithm: {algorithm}")
                return None
        except Exception as e:
            self.logger.error(f"Error creating algorithm instance: {str(e)}")
            return None

    async def cancel_algorithm(self, algorithm_id: str) -> bool:
        """
        Cancel an active execution algorithm.

        Args:
            algorithm_id: ID of the algorithm to cancel

        Returns:
            True if cancellation was successful, False otherwise
        """
        if algorithm_id not in self.active_algorithms:
            self.logger.warning(f"Algorithm {algorithm_id} not found or already completed")
            return False

        try:
            # Get the algorithm instance
            algo_instance = self.active_algorithms[algorithm_id]

            # Cancel the algorithm
            self.logger.info(f"Cancelling algorithm {algorithm_id}")
            return await algo_instance.cancel()
        except Exception as e:
            self.logger.error(f"Error cancelling algorithm {algorithm_id}: {str(e)}")
            return False

    def get_algorithm_status(self, algorithm_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of an active execution algorithm.

        Args:
            algorithm_id: ID of the algorithm to get status for

        Returns:
            Dictionary with algorithm status, or None if not found
        """
        if algorithm_id not in self.active_algorithms:
            return None

        try:
            # Get the algorithm instance
            algo_instance = self.active_algorithms[algorithm_id]

            # Get the status
            return asyncio.run(algo_instance.get_status())
        except Exception as e:
            self.logger.error(f"Error getting algorithm status for {algorithm_id}: {str(e)}")
            return None

    def get_active_algorithms(self) -> List[str]:
        """
        Get a list of active algorithm IDs.

        Returns:
            List of active algorithm IDs
        """
        return list(self.active_algorithms.keys())
