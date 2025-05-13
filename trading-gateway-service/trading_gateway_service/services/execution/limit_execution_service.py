"""
Limit Execution Service.

This module provides a specialized service for executing limit orders.
"""
import logging
import asyncio
import uuid
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from ...interfaces.broker_adapter_interface import BrokerAdapterInterface, OrderRequest, OrderType, ExecutionReport, OrderStatus
from .base_execution_service import BaseExecutionService
from .execution_mode_handler import ExecutionModeHandler, ExecutionMode


from trading_gateway_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class LimitExecutionService(BaseExecutionService):
    """
    Service for executing limit orders.

    Handles the execution of limit orders through broker adapters,
    with support for different execution modes.
    """

    def __init__(self, broker_adapters: Dict[str, BrokerAdapterInterface],
        mode_handler: ExecutionModeHandler, logger: Optional[logging.Logger
        ]=None):
        """
        Initialize the limit execution service.

        Args:
            broker_adapters: Dictionary of broker adapters by name
            mode_handler: Handler for different execution modes
            logger: Logger instance
        """
        super().__init__(broker_adapters, mode_handler, logger)
        self.logger.info('LimitExecutionService initialized')

    @with_exception_handling
    def place_order(self, order: OrderRequest, broker_name: Optional[str]=
        None, **kwargs) ->ExecutionReport:
        """
        Place a limit order with a specified broker.

        Args:
            order: Order request to be placed
            broker_name: Name of the broker to use, or None for default
            **kwargs: Additional arguments specific to limit orders

        Returns:
            Execution report of the order placement
        """
        if order.order_type != OrderType.LIMIT:
            error_msg = (
                f'Invalid order type for LimitExecutionService: {order.order_type}'
                )
            self.logger.error(error_msg)
            return ExecutionReport(order_id=str(uuid.uuid4()),
                client_order_id=order.client_order_id, instrument=order.
                instrument, status=OrderStatus.REJECTED, direction=order.
                direction, order_type=order.order_type, quantity=order.
                quantity, filled_quantity=0.0, price=order.price,
                rejection_reason=error_msg)
        if order.price is None:
            error_msg = 'Limit price must be specified for limit orders'
            self.logger.error(error_msg)
            return ExecutionReport(order_id=str(uuid.uuid4()),
                client_order_id=order.client_order_id, instrument=order.
                instrument, status=OrderStatus.REJECTED, direction=order.
                direction, order_type=order.order_type, quantity=order.
                quantity, filled_quantity=0.0, price=None, rejection_reason
                =error_msg)
        mode_report = self.mode_handler.handle_order_placement(order)
        if mode_report is not None:
            order_id = mode_report.order_id
            self.orders[order_id] = {'order': order, 'broker': broker_name or
                self.default_broker, 'execution_report': mode_report,
                'timestamp': datetime.utcnow()}
            self._trigger_callbacks('order_placed', self.orders[order_id])
            return mode_report
        if broker_name is None:
            broker_name = self.default_broker
        if broker_name not in self.broker_adapters:
            error_msg = f"Broker '{broker_name}' not registered"
            self.logger.error(error_msg)
            return ExecutionReport(order_id=str(uuid.uuid4()),
                client_order_id=order.client_order_id, instrument=order.
                instrument, status=OrderStatus.REJECTED, direction=order.
                direction, order_type=order.order_type, quantity=order.
                quantity, filled_quantity=0.0, price=order.price,
                rejection_reason=error_msg)
        broker_adapter = self.broker_adapters[broker_name]
        try:
            place_order_result = broker_adapter.place_order(order)
            if asyncio.iscoroutine(place_order_result):
                execution_report = asyncio.run(place_order_result)
            else:
                execution_report = place_order_result
            order_id = execution_report.order_id
            self.orders[order_id] = {'order': order, 'broker': broker_name,
                'execution_report': execution_report, 'timestamp': datetime
                .utcnow()}
            self._trigger_callbacks('order_placed', self.orders[order_id])
            if execution_report.status == OrderStatus.FILLED:
                self._trigger_callbacks('order_filled', self.orders[order_id])
            elif execution_report.status == OrderStatus.REJECTED:
                self._trigger_callbacks('order_rejected', self.orders[order_id]
                    )
            return execution_report
        except Exception as e:
            rejection_report = self._handle_execution_error(error=e,
                client_order_id=order.client_order_id, instrument=order.
                instrument, direction=order.direction, order_type=order.
                order_type, quantity=order.quantity, price=order.price,
                error_prefix='Error placing limit order')
            order_id = rejection_report.order_id
            self.orders[order_id] = {'order': order, 'broker': broker_name,
                'execution_report': rejection_report, 'timestamp': datetime
                .utcnow(), 'error': str(e)}
            self._trigger_callbacks('order_rejected', self.orders[order_id])
            return rejection_report

    @with_exception_handling
    def cancel_order(self, order_id: str) ->ExecutionReport:
        """
        Cancel an existing limit order.

        Args:
            order_id: ID of the order to cancel

        Returns:
            Execution report of the cancellation
        """
        if order_id not in self.orders:
            error_msg = f'Order ID {order_id} not found'
            self.logger.error(error_msg)
            return ExecutionReport(order_id=order_id, client_order_id=
                'unknown', instrument='unknown', status=OrderStatus.
                REJECTED, direction=None, order_type=None, quantity=0.0,
                filled_quantity=0.0, price=None, rejection_reason=error_msg)
        order_info = self.orders[order_id]
        broker_name = order_info['broker']
        execution_report = order_info['execution_report']
        if execution_report.status in [OrderStatus.FILLED, OrderStatus.
            CANCELLED, OrderStatus.REJECTED]:
            error_msg = (
                f'Cannot cancel order with status {execution_report.status}')
            self.logger.error(error_msg)
            return ExecutionReport(order_id=order_id, client_order_id=
                execution_report.client_order_id, instrument=
                execution_report.instrument, status=OrderStatus.REJECTED,
                direction=execution_report.direction, order_type=
                execution_report.order_type, quantity=execution_report.
                quantity, filled_quantity=execution_report.filled_quantity,
                price=execution_report.price, rejection_reason=error_msg)
        mode_report = self.mode_handler.handle_order_cancellation(order_id,
            order_info)
        if mode_report is not None:
            self.orders[order_id]['execution_report'] = mode_report
            self._trigger_callbacks('order_cancelled', self.orders[order_id])
            return mode_report
        broker_adapter = self.broker_adapters[broker_name]
        try:
            cancellation_report = asyncio.run(broker_adapter.cancel_order(
                order_id))
            self.orders[order_id]['execution_report'] = cancellation_report
            self._trigger_callbacks('order_cancelled', self.orders[order_id])
            return cancellation_report
        except Exception as e:
            rejection_report = self._handle_execution_error(error=e,
                order_id=order_id, client_order_id=execution_report.
                client_order_id, instrument=execution_report.instrument,
                direction=execution_report.direction, order_type=
                execution_report.order_type, quantity=execution_report.
                quantity, price=execution_report.price, error_prefix=
                'Error cancelling order')
            return rejection_report

    @with_exception_handling
    def modify_order(self, order_id: str, modifications: Dict[str, Any]
        ) ->ExecutionReport:
        """
        Modify an existing limit order.

        Args:
            order_id: ID of the order to modify
            modifications: Dictionary of parameters to modify

        Returns:
            Execution report of the modification
        """
        if order_id not in self.orders:
            error_msg = f'Order ID {order_id} not found'
            self.logger.error(error_msg)
            return ExecutionReport(order_id=order_id, client_order_id=
                'unknown', instrument='unknown', status=OrderStatus.
                REJECTED, direction=None, order_type=None, quantity=0.0,
                filled_quantity=0.0, price=None, rejection_reason=error_msg)
        order_info = self.orders[order_id]
        broker_name = order_info['broker']
        execution_report = order_info['execution_report']
        if execution_report.status in [OrderStatus.FILLED, OrderStatus.
            CANCELLED, OrderStatus.REJECTED]:
            error_msg = (
                f'Cannot modify order with status {execution_report.status}')
            self.logger.error(error_msg)
            return ExecutionReport(order_id=order_id, client_order_id=
                execution_report.client_order_id, instrument=
                execution_report.instrument, status=OrderStatus.REJECTED,
                direction=execution_report.direction, order_type=
                execution_report.order_type, quantity=execution_report.
                quantity, filled_quantity=execution_report.filled_quantity,
                price=execution_report.price, rejection_reason=error_msg)
        mode_report = self.mode_handler.handle_order_modification(order_id,
            order_info, modifications)
        if mode_report is not None:
            self.orders[order_id]['execution_report'] = mode_report
            self._trigger_callbacks('order_modified', self.orders[order_id])
            return mode_report
        broker_adapter = self.broker_adapters[broker_name]
        try:
            modification_report = asyncio.run(broker_adapter.modify_order(
                order_id, modifications))
            self.orders[order_id]['execution_report'] = modification_report
            self._trigger_callbacks('order_modified', self.orders[order_id])
            return modification_report
        except Exception as e:
            rejection_report = self._handle_execution_error(error=e,
                order_id=order_id, client_order_id=execution_report.
                client_order_id, instrument=execution_report.instrument,
                direction=execution_report.direction, order_type=
                execution_report.order_type, quantity=execution_report.
                quantity, price=execution_report.price, error_prefix=
                'Error modifying order')
            return rejection_report
