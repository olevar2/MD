"""
Unit tests for the refactored OrderExecutionService.

This module contains tests for the refactored OrderExecutionService class,
ensuring that it maintains backward compatibility with the original implementation.
"""

import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

# Add the trading-gateway-service directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from trading_gateway_service.interfaces.broker_adapter_interface import (
    OrderRequest,
    OrderStatus,
    OrderDirection,
    OrderType,
    ExecutionReport,
)
from trading_gateway_service.services.order_execution_service import (
    OrderExecutionService,
)
from trading_gateway_service.services.execution import (
    ExecutionMode,
    ExecutionAlgorithm,
)
from tests.services.mock_broker_adapter_interface import MockBrokerAdapterInterface


class MockBrokerAdapter(MockBrokerAdapterInterface):
    """Mock broker adapter for testing."""

    def __init__(self, name: str):
        """Initialize the mock broker adapter."""
        self.name = name
        self.connected = False
        self.orders = []

    def is_connected(self) -> bool:
        """Check if the broker is connected."""
        return self.connected

    def connect(self, credentials=None) -> bool:
        """Connect to the broker."""
        self.connected = True
        return True

    def disconnect(self) -> bool:
        """Disconnect from the broker."""
        self.connected = False
        return True

    def place_order(self, order_request):
        """Place an order with the broker."""
        self.orders.append(order_request)

        # Create a mock execution report
        return ExecutionReport(
            order_id=f"{self.name}-{len(self.orders)}",
            client_order_id=order_request.client_order_id,
            instrument=order_request.instrument,
            status=OrderStatus.OPEN,
            direction=order_request.direction,
            order_type=order_request.order_type,
            quantity=order_request.quantity,
            filled_quantity=0.0,
            price=order_request.price,
        )

    def cancel_order(self, order_id):
        """Cancel an order with the broker."""
        # Create a mock cancellation report
        return ExecutionReport(
            order_id=order_id,
            client_order_id="test",
            instrument="EURUSD",
            status=OrderStatus.CANCELLED,
            direction=OrderDirection.BUY,
            order_type=OrderType.MARKET,
            quantity=10000,
            filled_quantity=0.0,
            price=1.1000,
        )

    def modify_order(self, order_id, modifications):
        """Modify an order with the broker."""
        # Create a mock modification report
        return ExecutionReport(
            order_id=order_id,
            client_order_id="test",
            instrument="EURUSD",
            status=OrderStatus.OPEN,
            direction=OrderDirection.BUY,
            order_type=OrderType.MARKET,
            quantity=modifications.get("quantity", 10000),
            filled_quantity=0.0,
            price=modifications.get("price", 1.1000),
        )

    def get_orders(self):
        """Get all orders from the broker."""
        return self.orders

    def get_positions(self):
        """Get all positions from the broker."""
        return []

    def get_account_info(self):
        """Get account information from the broker."""
        return {
            "account_id": f"{self.name}-account",
            "balance": 100000.0,
            "currency": "USD",
        }

    def get_broker_info(self):
        """Get broker information."""
        return {
            "name": self.name,
            "description": f"Mock broker adapter for {self.name}",
            "version": "1.0.0",
        }

    def get_market_data(self, instrument, data_type=None):
        """Get market data from the broker."""
        return {
            "instrument": instrument,
            "bid": 1.1000,
            "ask": 1.1001,
            "timestamp": datetime.utcnow(),
        }


class TestRefactoredOrderExecutionService(unittest.TestCase):
    """Tests for the refactored OrderExecutionService."""

    def setUp(self):
        """Set up test fixtures."""
        self.service = OrderExecutionService(mode=ExecutionMode.SIMULATED)

        # Register broker adapters
        self.broker1 = MockBrokerAdapter("broker1")
        self.broker2 = MockBrokerAdapter("broker2")
        self.broker3 = MockBrokerAdapter("broker3")

        self.service.register_broker_adapter("broker1", self.broker1, default=True)
        self.service.register_broker_adapter("broker2", self.broker2)
        self.service.register_broker_adapter("broker3", self.broker3)

        # Create test orders
        self.market_order = OrderRequest(
            instrument="EURUSD",
            order_type=OrderType.MARKET,
            direction=OrderDirection.BUY,
            quantity=10000,
            price=1.1000,
            client_order_id="test-market-order",
        )

        self.limit_order = OrderRequest(
            instrument="EURUSD",
            order_type=OrderType.LIMIT,
            direction=OrderDirection.BUY,
            quantity=10000,
            price=1.1000,
            client_order_id="test-limit-order",
        )

        self.stop_order = OrderRequest(
            instrument="EURUSD",
            order_type=OrderType.STOP,
            direction=OrderDirection.BUY,
            quantity=10000,
            price=1.1000,
            client_order_id="test-stop-order",
        )

    def test_register_broker_adapter(self):
        """Test registering a broker adapter."""
        # Register a new broker adapter
        broker4 = MockBrokerAdapter("broker4")
        self.service.register_broker_adapter("broker4", broker4)

        # Check that the broker adapter was registered
        self.assertIn("broker4", self.service.broker_adapters)

        # Check that the default broker is still broker1
        self.assertEqual(self.service.default_broker, "broker1")

        # Register a new broker adapter as default
        broker5 = MockBrokerAdapter("broker5")
        self.service.register_broker_adapter("broker5", broker5, default=True)

        # Check that the broker adapter was registered
        self.assertIn("broker5", self.service.broker_adapters)

        # Check that the default broker is now broker5
        self.assertEqual(self.service.default_broker, "broker5")

    def test_set_default_broker(self):
        """Test setting the default broker."""
        # Set broker2 as default
        self.service.default_broker = "broker2"

        # Check that the default broker is now broker2
        self.assertEqual(self.service.default_broker, "broker2")

    def test_connect(self):
        """Test connecting to brokers."""
        # Connect to all brokers
        result = self.service.connect()

        # Check that the operation was successful
        self.assertTrue(result)

        # Check that all brokers are connected
        self.assertTrue(self.broker1.is_connected())
        self.assertTrue(self.broker2.is_connected())
        self.assertTrue(self.broker3.is_connected())

        # Disconnect all brokers
        self.broker1.connected = False
        self.broker2.connected = False
        self.broker3.connected = False

        # Connect to a specific broker
        result = self.service.connect("broker1")

        # Check that the operation was successful
        self.assertTrue(result)

        # Check that only broker1 is connected
        self.assertTrue(self.broker1.is_connected())
        self.assertFalse(self.broker2.is_connected())
        self.assertFalse(self.broker3.is_connected())

        # Try to connect to a non-existent broker
        result = self.service.connect("non-existent")

        # Check that the operation failed
        self.assertFalse(result)

    def test_disconnect(self):
        """Test disconnecting from brokers."""
        # Connect all brokers
        self.broker1.connected = True
        self.broker2.connected = True
        self.broker3.connected = True

        # Disconnect from all brokers
        result = self.service.disconnect()

        # Check that the operation was successful
        self.assertTrue(result)

        # Check that all brokers are disconnected
        self.assertFalse(self.broker1.is_connected())
        self.assertFalse(self.broker2.is_connected())
        self.assertFalse(self.broker3.is_connected())

        # Connect all brokers again
        self.broker1.connected = True
        self.broker2.connected = True
        self.broker3.connected = True

        # Disconnect from a specific broker
        result = self.service.disconnect("broker1")

        # Check that the operation was successful
        self.assertTrue(result)

        # Check that only broker1 is disconnected
        self.assertFalse(self.broker1.is_connected())
        self.assertTrue(self.broker2.is_connected())
        self.assertTrue(self.broker3.is_connected())

        # Try to disconnect from a non-existent broker
        result = self.service.disconnect("non-existent")

        # Check that the operation failed
        self.assertFalse(result)

    def test_place_market_order(self):
        """Test placing a market order."""
        # Place a market order
        report = self.service.place_order(self.market_order)

        # Check that the order was placed
        self.assertEqual(report.client_order_id, self.market_order.client_order_id)
        self.assertEqual(report.instrument, self.market_order.instrument)
        self.assertEqual(report.status, OrderStatus.OPEN)

        # Check that the order was placed with the default broker
        self.assertEqual(len(self.broker1.orders), 1)
        self.assertEqual(self.broker1.orders[0].client_order_id, self.market_order.client_order_id)

        # Place a market order with a specific broker
        report = self.service.place_order(self.market_order, broker_name="broker2")

        # Check that the order was placed
        self.assertEqual(report.client_order_id, self.market_order.client_order_id)
        self.assertEqual(report.instrument, self.market_order.instrument)
        self.assertEqual(report.status, OrderStatus.OPEN)

        # Check that the order was placed with broker2
        self.assertEqual(len(self.broker2.orders), 1)
        self.assertEqual(self.broker2.orders[0].client_order_id, self.market_order.client_order_id)

        # Try to place a market order with a non-existent broker
        report = self.service.place_order(self.market_order, broker_name="non-existent")

        # Check that the order was rejected
        self.assertEqual(report.status, OrderStatus.REJECTED)

    def test_place_limit_order(self):
        """Test placing a limit order."""
        # Place a limit order
        report = self.service.place_order(self.limit_order)

        # Check that the order was placed
        self.assertEqual(report.client_order_id, self.limit_order.client_order_id)
        self.assertEqual(report.instrument, self.limit_order.instrument)
        self.assertEqual(report.status, OrderStatus.OPEN)

        # Check that the order was placed with the default broker
        self.assertEqual(len(self.broker1.orders), 1)
        self.assertEqual(self.broker1.orders[0].client_order_id, self.limit_order.client_order_id)

    def test_place_stop_order(self):
        """Test placing a stop order."""
        # Place a stop order
        report = self.service.place_order(self.stop_order)

        # Check that the order was placed
        self.assertEqual(report.client_order_id, self.stop_order.client_order_id)
        self.assertEqual(report.instrument, self.stop_order.instrument)
        self.assertEqual(report.status, OrderStatus.OPEN)

        # Check that the order was placed with the default broker
        self.assertEqual(len(self.broker1.orders), 1)
        self.assertEqual(self.broker1.orders[0].client_order_id, self.stop_order.client_order_id)

    def test_cancel_order(self):
        """Test cancelling an order."""
        # Place an order
        report = self.service.place_order(self.market_order)
        order_id = report.order_id

        # Cancel the order
        cancel_report = self.service.cancel_order(order_id)

        # Check that the order was cancelled
        self.assertEqual(cancel_report.order_id, order_id)
        self.assertEqual(cancel_report.status, OrderStatus.CANCELLED)

        # Try to cancel a non-existent order
        cancel_report = self.service.cancel_order("non-existent")

        # Check that the cancellation was rejected
        self.assertEqual(cancel_report.status, OrderStatus.REJECTED)

    def test_modify_order(self):
        """Test modifying an order."""
        # Place an order
        report = self.service.place_order(self.market_order)
        order_id = report.order_id

        # Modify the order
        modifications = {"quantity": 20000, "price": 1.2000}
        modify_report = self.service.modify_order(order_id, modifications)

        # Check that the order was modified
        self.assertEqual(modify_report.order_id, order_id)
        self.assertEqual(modify_report.quantity, 20000)
        self.assertEqual(modify_report.price, 1.2000)

        # Try to modify a non-existent order
        modify_report = self.service.modify_order("non-existent", modifications)

        # Check that the modification was rejected
        self.assertEqual(modify_report.status, OrderStatus.REJECTED)

    def test_get_orders(self):
        """Test getting orders."""
        # Place some orders
        self.service.place_order(self.market_order)
        self.service.place_order(self.limit_order)
        self.service.place_order(self.stop_order)

        # Get all orders
        orders = self.service.get_orders()

        # Check that all orders were returned
        self.assertEqual(len(orders), 3)

        # Get orders for a specific instrument
        orders = self.service.get_orders(instrument="EURUSD")

        # Check that all orders were returned (all are EURUSD)
        self.assertEqual(len(orders), 3)

        # Get orders with a specific status
        orders = self.service.get_orders(status=OrderStatus.OPEN)

        # Check that all orders were returned (all are OPEN)
        self.assertEqual(len(orders), 3)

        # Get orders for a specific instrument and status
        orders = self.service.get_orders(instrument="EURUSD", status=OrderStatus.OPEN)

        # Check that all orders were returned (all are EURUSD and OPEN)
        self.assertEqual(len(orders), 3)

        # Get orders for a non-existent instrument
        orders = self.service.get_orders(instrument="GBPUSD")

        # Check that no orders were returned
        self.assertEqual(len(orders), 0)

        # Get orders with a non-existent status
        orders = self.service.get_orders(status=OrderStatus.FILLED)

        # Check that no orders were returned
        self.assertEqual(len(orders), 0)

    def test_get_order(self):
        """Test getting a specific order."""
        # Place an order
        report = self.service.place_order(self.market_order)
        order_id = report.order_id

        # Get the order
        order = self.service.get_order(order_id)

        # Check that the order was returned
        self.assertIsNotNone(order)
        self.assertEqual(order["execution_report"].order_id, order_id)

        # Try to get a non-existent order
        order = self.service.get_order("non-existent")

        # Check that no order was returned
        self.assertIsNone(order)

    def test_update_execution_status(self):
        """Test updating the execution status of an order."""
        # Place an order
        report = self.service.place_order(self.market_order)
        order_id = report.order_id

        # Update the execution status
        status_update = {"status": OrderStatus.FILLED, "filled_quantity": 10000, "executed_price": 1.1000}
        result = self.service.update_execution_status(order_id, status_update)

        # Check that the update was successful
        self.assertTrue(result)

        # Get the order
        order = self.service.get_order(order_id)

        # Check that the order status was updated
        self.assertEqual(order["execution_report"].status, OrderStatus.FILLED)
        self.assertEqual(order["execution_report"].filled_quantity, 10000)
        self.assertEqual(order["execution_report"].executed_price, 1.1000)

        # Try to update a non-existent order
        result = self.service.update_execution_status("non-existent", status_update)

        # Check that the update failed
        self.assertFalse(result)

    def test_algorithm_execution(self):
        """Test executing an order with an algorithm."""
        # Mock the algorithm execution service
        self.service.algorithm_execution_service = MagicMock()

        # Create a mock execution report
        mock_report = ExecutionReport(
            order_id="algo-1",
            client_order_id=self.market_order.client_order_id,
            instrument=self.market_order.instrument,
            status=OrderStatus.FILLED,
            direction=self.market_order.direction,
            order_type=self.market_order.order_type,
            quantity=self.market_order.quantity,
            filled_quantity=self.market_order.quantity,
            price=self.market_order.price,
            executed_price=self.market_order.price,
        )

        # Set up the mock to return our mock report
        self.service.algorithm_execution_service.place_order.return_value = mock_report

        # Place an order with the SOR algorithm
        report = self.service.place_order(
            self.market_order,
            algorithm=ExecutionAlgorithm.SOR,
            algorithm_config={"min_brokers": 2}
        )

        # Check that the algorithm execution service was called
        self.service.algorithm_execution_service.place_order.assert_called_once()

        # Check that the order was executed
        self.assertEqual(report.client_order_id, self.market_order.client_order_id)
        self.assertEqual(report.instrument, self.market_order.instrument)
        self.assertEqual(report.status, OrderStatus.FILLED)

    def test_get_algorithm_status(self):
        """Test getting the status of an algorithm."""
        # Mock the algorithm execution service
        self.service.algorithm_execution_service = MagicMock()

        # Set up the mock to return a status
        self.service.algorithm_execution_service.get_algorithm_status.return_value = {
            "algorithm_id": "algo-1",
            "status": "running",
            "progress": 0.5,
            "filled_quantity": 5000,
            "remaining_quantity": 5000,
        }

        # Get the status of an algorithm
        status = self.service.get_algorithm_status("algo-1")

        # Check that the algorithm execution service was called
        self.service.algorithm_execution_service.get_algorithm_status.assert_called_once_with("algo-1")

        # Check that the status was returned
        self.assertIsNotNone(status)
        self.assertEqual(status["status"], "running")
        self.assertEqual(status["progress"], 0.5)

        # Mock the get_algorithm_status method to return None
        self.service.algorithm_execution_service.get_algorithm_status.return_value = None

        # Try to get the status of a non-existent algorithm
        status = self.service.get_algorithm_status("non-existent")

        # Check that no status was returned
        self.assertIsNone(status)

    def test_get_active_algorithms(self):
        """Test getting active algorithms."""
        # Mock the algorithm execution service
        self.service.algorithm_execution_service = MagicMock()

        # Set up the mock to return a list of active algorithms
        self.service.algorithm_execution_service.get_active_algorithms.return_value = ["algo-1", "algo-2"]

        # Get active algorithms
        algorithms = self.service.get_active_algorithms()

        # Check that the algorithm execution service was called
        self.service.algorithm_execution_service.get_active_algorithms.assert_called_once()

        # Check that the algorithms were returned
        self.assertEqual(len(algorithms), 2)
        self.assertIn("algo-1", algorithms)
        self.assertIn("algo-2", algorithms)

        # Mock the get_active_algorithms method to return an empty list
        self.service.algorithm_execution_service.get_active_algorithms.return_value = []

        # Get active algorithms when there are none
        algorithms = self.service.get_active_algorithms()

        # Check that no algorithms were returned
        self.assertEqual(len(algorithms), 0)


if __name__ == "__main__":
    print("Running tests for refactored OrderExecutionService...")
    result = unittest.main(exit=False)
    print(f"Tests completed with result: {result.result}")