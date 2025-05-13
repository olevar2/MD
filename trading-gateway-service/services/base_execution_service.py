"""
Base Execution Service.

This module provides the base class for all execution services, defining
the common interface and shared functionality.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
import uuid
from ...interfaces.broker_adapter_interface import BrokerAdapterInterface, OrderRequest, ExecutionReport, OrderStatus
from .execution_mode_handler import ExecutionModeHandler, ExecutionMode
from core.exceptions_bridge_1 import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from utils.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class BaseExecutionService(ABC):
    """
    Base class for all execution services.

    Defines the common interface and shared functionality for executing orders
    through broker adapters.
    """

    def __init__(self, broker_adapters: Dict[str, BrokerAdapterInterface],
        mode_handler: ExecutionModeHandler, logger: Optional[logging.Logger
        ]=None):
        """
        Initialize the base execution service.

        Args:
            broker_adapters: Dictionary of broker adapters by name
            mode_handler: Handler for different execution modes
            logger: Logger instance
        """
        self.broker_adapters = broker_adapters
        self.mode_handler = mode_handler
        self.logger = logger or logging.getLogger(__name__)
        self.default_broker = None
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.callbacks: Dict[str, List[Callable]] = {'order_placed': [],
            'order_filled': [], 'order_cancelled': [], 'order_rejected': [],
            'order_modified': [], 'execution_error': []}
        for broker_name, adapter in broker_adapters.items():
            if self.default_broker is None:
                self.default_broker = broker_name

    def set_default_broker(self, broker_name: str) ->bool:
        """
        Set the default broker for order execution.

        Args:
            broker_name: Name of the broker to set as default

        Returns:
            True if successful, False if broker not found
        """
        if broker_name not in self.broker_adapters:
            self.logger.error(
                f"Cannot set default broker: '{broker_name}' not registered")
            return False
        self.default_broker = broker_name
        self.logger.info(f"Default broker set to '{broker_name}'")
        return True

    def register_callback(self, event_type: str, callback: Callable) ->bool:
        """
        Register a callback for a specific event type.

        Args:
            event_type: Type of event to register for
            callback: Callback function to register

        Returns:
            True if registration successful, False otherwise
        """
        if event_type not in self.callbacks:
            self.logger.error(f'Unknown event type: {event_type}')
            return False
        self.callbacks[event_type].append(callback)
        return True

    @with_exception_handling
    def _trigger_callbacks(self, event_type: str, data: Any) ->None:
        """
        Trigger all callbacks for a specific event type.

        Args:
            event_type: Type of event that occurred
            data: Data to pass to the callbacks
        """
        if event_type not in self.callbacks:
            return
        for callback in self.callbacks[event_type]:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f'Error in {event_type} callback: {str(e)}')

    def _handle_execution_error(self, error: Exception, order_id: str=None,
        client_order_id: str=None, instrument: str=None, direction=None,
        order_type=None, quantity: float=0.0, price=None, error_prefix: str
        ='Error in execution') ->ExecutionReport:
        """
        Handle an error during order execution and create a rejection report.

        Args:
            error: The exception that occurred
            order_id: ID of the order (if available)
            client_order_id: Client order ID (if available)
            instrument: Trading instrument (if available)
            direction: Order direction (if available)
            order_type: Order type (if available)
            quantity: Order quantity (if available)
            price: Order price (if available)
            error_prefix: Prefix for the error message

        Returns:
            Execution report with REJECTED status
        """
        error_msg = f'{error_prefix}: {str(error)}'
        self.logger.error(error_msg, exc_info=True)
        rejection_report = ExecutionReport(order_id=order_id or str(uuid.
            uuid4()), client_order_id=client_order_id or 'unknown',
            instrument=instrument or 'unknown', status=OrderStatus.REJECTED,
            direction=direction, order_type=order_type, quantity=quantity,
            filled_quantity=0.0, price=price, rejection_reason=error_msg)
        if order_id and order_id in self.orders:
            self.orders[order_id]['execution_report'] = rejection_report
            self.orders[order_id]['error'] = str(error)
            self._trigger_callbacks('execution_error', {'order_id':
                order_id, 'error': str(error), 'order_info': self.orders[
                order_id]})
        return rejection_report

    def _create_dummy_execution_report(self, order: OrderRequest, status:
        OrderStatus) ->ExecutionReport:
        """
        Create a dummy execution report for simulation or testing.

        Args:
            order: The order request
            status: The status to set in the report

        Returns:
            Execution report with the specified status
        """
        order_id = str(uuid.uuid4())
        executed_price = order.price if status == OrderStatus.FILLED else None
        filled_quantity = (order.quantity if status == OrderStatus.FILLED else
            0.0)
        return ExecutionReport(order_id=order_id, client_order_id=order.
            client_order_id, instrument=order.instrument, status=status,
            direction=order.direction, order_type=order.order_type,
            quantity=order.quantity, filled_quantity=filled_quantity, price
            =order.price, executed_price=executed_price, transaction_time=
            datetime.now(timezone.utc))

    @with_broker_api_resilience('get_orders')
    def get_orders(self, instrument: Optional[str]=None, status: Optional[
        OrderStatus]=None) ->List[Dict[str, Any]]:
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
            execution_report = order_info['execution_report']
            if instrument and execution_report.instrument != instrument:
                continue
            if status and execution_report.status != status:
                continue
            filtered_orders.append(order_info)
        return filtered_orders

    @with_broker_api_resilience('get_order')
    def get_order(self, order_id: str) ->Optional[Dict[str, Any]]:
        """
        Get information about a specific order.

        Args:
            order_id: ID of the order to retrieve

        Returns:
            Order information dictionary, or None if not found
        """
        return self.orders.get(order_id)

    @with_broker_api_resilience('update_execution_status')
    def update_execution_status(self, order_id: str, status_update: Dict[
        str, Any]) ->bool:
        """
        Update the status of an order (e.g., from external callbacks/websockets).

        Args:
            order_id: ID of the order to update
            status_update: New status information

        Returns:
            True if update was successful
        """
        if order_id not in self.orders:
            self.logger.warning(
                f'Cannot update status for unknown order ID: {order_id}')
            return False
        order_info = self.orders[order_id]
        execution_report = order_info['execution_report']
        for key, value in status_update.items():
            if hasattr(execution_report, key):
                setattr(execution_report, key, value)
        order_info['execution_report'] = execution_report
        self.orders[order_id] = order_info
        if 'status' in status_update:
            new_status = status_update['status']
            if new_status == OrderStatus.FILLED:
                self._trigger_callbacks('order_filled', order_info)
            elif new_status == OrderStatus.CANCELLED:
                self._trigger_callbacks('order_cancelled', order_info)
            elif new_status == OrderStatus.REJECTED:
                self._trigger_callbacks('order_rejected', order_info)
        return True

    @abstractmethod
    def place_order(self, order: OrderRequest, broker_name: Optional[str]=
        None, **kwargs) ->ExecutionReport:
        """
        Place an order with a specified broker.

        Args:
            order: Order request to be placed
            broker_name: Name of the broker to use, or None for default
            **kwargs: Additional arguments specific to the order type

        Returns:
            Execution report of the order placement
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) ->ExecutionReport:
        """
        Cancel an existing order.

        Args:
            order_id: ID of the order to cancel

        Returns:
            Execution report of the cancellation
        """
        pass

    @abstractmethod
    def modify_order(self, order_id: str, modifications: Dict[str, Any]
        ) ->ExecutionReport:
        """
        Modify an existing order.

        Args:
            order_id: ID of the order to modify
            modifications: Dictionary of parameters to modify

        Returns:
            Execution report of the modification
        """
        pass
