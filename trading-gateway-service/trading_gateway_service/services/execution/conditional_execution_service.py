"""
Conditional Execution Service.

This module provides a specialized service for executing conditional orders.
"""

import logging
import asyncio
import uuid
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from ...interfaces.broker_adapter_interface import (
    BrokerAdapterInterface,
    OrderRequest,
    OrderType,
    ExecutionReport,
    OrderStatus,
)
from .base_execution_service import BaseExecutionService
from .execution_mode_handler import ExecutionModeHandler, ExecutionMode


class ConditionalExecutionService(BaseExecutionService):
    """
    Service for executing conditional orders.
    
    Handles the execution of conditional orders through broker adapters,
    with support for different execution modes.
    """
    
    def __init__(self, 
                 broker_adapters: Dict[str, BrokerAdapterInterface],
                 mode_handler: ExecutionModeHandler,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the conditional execution service.
        
        Args:
            broker_adapters: Dictionary of broker adapters by name
            mode_handler: Handler for different execution modes
            logger: Logger instance
        """
        super().__init__(broker_adapters, mode_handler, logger)
        self.logger.info("ConditionalExecutionService initialized")
    
    def place_order(self, order: OrderRequest, broker_name: Optional[str] = None, **kwargs) -> ExecutionReport:
        """
        Place a conditional order with a specified broker.
        
        Args:
            order: Order request to be placed
            broker_name: Name of the broker to use, or None for default
            **kwargs: Additional arguments specific to conditional orders
            
        Returns:
            Execution report of the order placement
        """
        # Verify that this is a conditional order
        if order.order_type != OrderType.CONDITIONAL:
            error_msg = f"Invalid order type for ConditionalExecutionService: {order.order_type}"
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
        
        # Verify that the condition is specified
        if "condition" not in kwargs:
            error_msg = "Condition must be specified for conditional orders"
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
        
        # Check if the mode handler wants to handle this order
        mode_report = self.mode_handler.handle_order_placement(order)
        if mode_report is not None:
            # Mode handler has handled the order
            order_id = mode_report.order_id
            self.orders[order_id] = {
                "order": order,
                "broker": broker_name or self.default_broker,
                "execution_report": mode_report,
                "timestamp": datetime.utcnow(),
                "condition": kwargs["condition"],
            }
            self._trigger_callbacks("order_placed", self.orders[order_id])
            return mode_report
        
        # Use the default broker if none specified
        if broker_name is None:
            broker_name = self.default_broker
        
        # Check if the broker exists
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
        
        # Get the broker adapter
        broker_adapter = self.broker_adapters[broker_name]
        
        try:
            # Place the order with the broker
            # Add the condition to the order
            order_with_condition = order
            order_with_condition.extra_params = order_with_condition.extra_params or {}
            order_with_condition.extra_params["condition"] = kwargs["condition"]
            
            execution_report = asyncio.run(broker_adapter.place_order(order_with_condition))
            
            # Store the order information
            order_id = execution_report.order_id
            self.orders[order_id] = {
                "order": order,
                "broker": broker_name,
                "execution_report": execution_report,
                "timestamp": datetime.utcnow(),
                "condition": kwargs["condition"],
            }
            
            # Trigger callbacks
            self._trigger_callbacks("order_placed", self.orders[order_id])
            
            if execution_report.status == OrderStatus.FILLED:
                self._trigger_callbacks("order_filled", self.orders[order_id])
            elif execution_report.status == OrderStatus.REJECTED:
                self._trigger_callbacks("order_rejected", self.orders[order_id])
                
            return execution_report
            
        except Exception as e:
            error_msg = f"Error placing conditional order: {str(e)}"
            self.logger.error(error_msg)
            
            # Create a rejection report
            rejection_report = ExecutionReport(
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
            
            # Store the order information
            order_id = rejection_report.order_id
            self.orders[order_id] = {
                "order": order,
                "broker": broker_name,
                "execution_report": rejection_report,
                "timestamp": datetime.utcnow(),
                "condition": kwargs["condition"],
                "error": str(e),
            }
            
            # Trigger callbacks
            self._trigger_callbacks("order_rejected", self.orders[order_id])
            self._trigger_callbacks("execution_error", {
                "order_id": order_id,
                "error": str(e),
                "order_info": self.orders[order_id],
            })
            
            return rejection_report
    
    def cancel_order(self, order_id: str) -> ExecutionReport:
        """
        Cancel an existing conditional order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            Execution report of the cancellation
        """
        # Check if the order exists
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
        
        # Get the order information
        order_info = self.orders[order_id]
        broker_name = order_info["broker"]
        execution_report = order_info["execution_report"]
        
        # Check if the order can be cancelled
        if execution_report.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            error_msg = f"Cannot cancel order with status {execution_report.status}"
            self.logger.error(error_msg)
            return ExecutionReport(
                order_id=order_id,
                client_order_id=execution_report.client_order_id,
                instrument=execution_report.instrument,
                status=OrderStatus.REJECTED,
                direction=execution_report.direction,
                order_type=execution_report.order_type,
                quantity=execution_report.quantity,
                filled_quantity=execution_report.filled_quantity,
                price=execution_report.price,
                rejection_reason=error_msg,
            )
        
        # Check if the mode handler wants to handle this cancellation
        mode_report = self.mode_handler.handle_order_cancellation(order_id, order_info)
        if mode_report is not None:
            # Mode handler has handled the cancellation
            self.orders[order_id]["execution_report"] = mode_report
            self._trigger_callbacks("order_cancelled", self.orders[order_id])
            return mode_report
        
        # Get the broker adapter
        broker_adapter = self.broker_adapters[broker_name]
        
        try:
            # Cancel the order with the broker
            cancellation_report = asyncio.run(broker_adapter.cancel_order(order_id))
            
            # Update the order information
            self.orders[order_id]["execution_report"] = cancellation_report
            
            # Trigger callbacks
            self._trigger_callbacks("order_cancelled", self.orders[order_id])
            
            return cancellation_report
            
        except Exception as e:
            error_msg = f"Error cancelling order: {str(e)}"
            self.logger.error(error_msg)
            
            # Create a rejection report
            rejection_report = ExecutionReport(
                order_id=order_id,
                client_order_id=execution_report.client_order_id,
                instrument=execution_report.instrument,
                status=OrderStatus.REJECTED,
                direction=execution_report.direction,
                order_type=execution_report.order_type,
                quantity=execution_report.quantity,
                filled_quantity=execution_report.filled_quantity,
                price=execution_report.price,
                rejection_reason=error_msg,
            )
            
            # Update the order information
            self.orders[order_id]["execution_report"] = rejection_report
            self.orders[order_id]["error"] = str(e)
            
            # Trigger callbacks
            self._trigger_callbacks("execution_error", {
                "order_id": order_id,
                "error": str(e),
                "order_info": self.orders[order_id],
            })
            
            return rejection_report
    
    def modify_order(self, order_id: str, modifications: Dict[str, Any]) -> ExecutionReport:
        """
        Modify an existing conditional order.
        
        Args:
            order_id: ID of the order to modify
            modifications: Dictionary of parameters to modify
            
        Returns:
            Execution report of the modification
        """
        # Check if the order exists
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
        
        # Get the order information
        order_info = self.orders[order_id]
        broker_name = order_info["broker"]
        execution_report = order_info["execution_report"]
        
        # Check if the order can be modified
        if execution_report.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            error_msg = f"Cannot modify order with status {execution_report.status}"
            self.logger.error(error_msg)
            return ExecutionReport(
                order_id=order_id,
                client_order_id=execution_report.client_order_id,
                instrument=execution_report.instrument,
                status=OrderStatus.REJECTED,
                direction=execution_report.direction,
                order_type=execution_report.order_type,
                quantity=execution_report.quantity,
                filled_quantity=execution_report.filled_quantity,
                price=execution_report.price,
                rejection_reason=error_msg,
            )
        
        # Check if the mode handler wants to handle this modification
        mode_report = self.mode_handler.handle_order_modification(order_id, order_info, modifications)
        if mode_report is not None:
            # Mode handler has handled the modification
            self.orders[order_id]["execution_report"] = mode_report
            self._trigger_callbacks("order_modified", self.orders[order_id])
            return mode_report
        
        # Get the broker adapter
        broker_adapter = self.broker_adapters[broker_name]
        
        try:
            # If the condition is being modified, add it to the modifications
            if "condition" in modifications:
                self.orders[order_id]["condition"] = modifications["condition"]
            
            # Modify the order with the broker
            modification_report = asyncio.run(broker_adapter.modify_order(order_id, modifications))
            
            # Update the order information
            self.orders[order_id]["execution_report"] = modification_report
            
            # Trigger callbacks
            self._trigger_callbacks("order_modified", self.orders[order_id])
            
            return modification_report
            
        except Exception as e:
            error_msg = f"Error modifying order: {str(e)}"
            self.logger.error(error_msg)
            
            # Create a rejection report
            rejection_report = ExecutionReport(
                order_id=order_id,
                client_order_id=execution_report.client_order_id,
                instrument=execution_report.instrument,
                status=OrderStatus.REJECTED,
                direction=execution_report.direction,
                order_type=execution_report.order_type,
                quantity=execution_report.quantity,
                filled_quantity=execution_report.filled_quantity,
                price=execution_report.price,
                rejection_reason=error_msg,
            )
            
            # Update the order information
            self.orders[order_id]["execution_report"] = rejection_report
            self.orders[order_id]["error"] = str(e)
            
            # Trigger callbacks
            self._trigger_callbacks("execution_error", {
                "order_id": order_id,
                "error": str(e),
                "order_info": self.orders[order_id],
            })
            
            return rejection_report