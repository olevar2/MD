"""
Unit tests for the OrderExecutionService.

This module contains tests for the OrderExecutionService class, including
tests for the execution algorithm integration.
"""

import asyncio
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
from typing import Dict, List, Any

from trading_gateway_service.interfaces.broker_adapter_interface import (
    OrderRequest,
    OrderStatus,
    OrderDirection,
    OrderType,
    ExecutionReport,
    BrokerAdapterInterface,
)
from trading_gateway_service.services.order_execution_service import (
    OrderExecutionService,
    ExecutionMode,
    ExecutionAlgorithm,
)
from trading_gateway_service.execution_algorithms import (
    BaseExecutionAlgorithm,
    SmartOrderRoutingAlgorithm,
    TWAPAlgorithm,
    VWAPAlgorithm,
    ImplementationShortfallAlgorithm,
)


class MockBrokerAdapter:
    """Mock broker adapter for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.orders = []
        self.connected = True
        
    def is_connected(self) -> bool:
        return self.connected
        
    def place_order(self, order: OrderRequest) -> ExecutionReport:
        """Mock place_order method."""
        self.orders.append(order)
        
        # Create a mock execution report
        return ExecutionReport(
            order_id=f"order-{len(self.orders)}",
            client_order_id=order.client_order_id,
            instrument=order.instrument,
            status=OrderStatus.FILLED,
            direction=order.direction,
            order_type=order.order_type,
            quantity=order.quantity,
            filled_quantity=order.quantity,
            price=order.price,
            executed_price=order.price,
        )
        
    def connect(self, credentials: Dict[str, str] = None) -> bool:
        """Mock connect method."""
        self.connected = True
        return True
        
    def disconnect(self) -> bool:
        """Mock disconnect method."""
        self.connected = False
        return True


class TestOrderExecutionService(unittest.TestCase):
    """Tests for the OrderExecutionService class."""
    
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
        
        # Create a test order
        self.order = OrderRequest(
            instrument="EURUSD",
            order_type=OrderType.MARKET,
            direction=OrderDirection.BUY,
            quantity=10000,
            price=1.1000,
            client_order_id="test-order-1",
        )
    
    def test_direct_execution(self):
        """Test direct order execution."""
        # Place the order with direct execution
        report = self.service.place_order(self.order, broker_name="broker1")
        
        # Check the execution report
        self.assertEqual(report.client_order_id, self.order.client_order_id)
        self.assertEqual(report.instrument, self.order.instrument)
        self.assertEqual(report.direction, self.order.direction)
        self.assertEqual(report.quantity, self.order.quantity)
        
        # Check that the order was stored
        self.assertTrue(report.order_id in self.service.orders)
        
    def test_sor_algorithm(self):
        """Test execution with the SOR algorithm."""
        # Mock the _execute_with_algorithm method
        original_method = self.service._execute_with_algorithm
        
        async def mock_execute_with_algorithm(order, algorithm, algorithm_config):
            # Create a mock execution result
            return ExecutionReport(
                order_id="algo-1",
                client_order_id=order.client_order_id,
                instrument=order.instrument,
                status=OrderStatus.FILLED,
                direction=order.direction,
                order_type=order.order_type,
                quantity=order.quantity,
                filled_quantity=order.quantity,
                price=order.price,
                executed_price=order.price,
            )
            
        self.service._execute_with_algorithm = mock_execute_with_algorithm
        
        # Place the order with the SOR algorithm
        report = self.service.place_order(
            self.order,
            algorithm=ExecutionAlgorithm.SOR,
            algorithm_config={"min_brokers": 2}
        )
        
        # Check the execution report
        self.assertEqual(report.client_order_id, self.order.client_order_id)
        self.assertEqual(report.instrument, self.order.instrument)
        self.assertEqual(report.status, OrderStatus.FILLED)
        
        # Restore the original method
        self.service._execute_with_algorithm = original_method
    
    def test_twap_algorithm(self):
        """Test execution with the TWAP algorithm."""
        # Mock the _execute_with_algorithm method
        original_method = self.service._execute_with_algorithm
        
        async def mock_execute_with_algorithm(order, algorithm, algorithm_config):
            # Create a mock execution result
            return ExecutionReport(
                order_id="algo-1",
                client_order_id=order.client_order_id,
                instrument=order.instrument,
                status=OrderStatus.FILLED,
                direction=order.direction,
                order_type=order.order_type,
                quantity=order.quantity,
                filled_quantity=order.quantity,
                price=order.price,
                executed_price=order.price,
            )
            
        self.service._execute_with_algorithm = mock_execute_with_algorithm
        
        # Place the order with the TWAP algorithm
        report = self.service.place_order(
            self.order,
            algorithm=ExecutionAlgorithm.TWAP,
            algorithm_config={"duration_minutes": 30, "num_slices": 6}
        )
        
        # Check the execution report
        self.assertEqual(report.client_order_id, self.order.client_order_id)
        self.assertEqual(report.instrument, self.order.instrument)
        self.assertEqual(report.status, OrderStatus.FILLED)
        
        # Restore the original method
        self.service._execute_with_algorithm = original_method
    
    def test_vwap_algorithm(self):
        """Test execution with the VWAP algorithm."""
        # Mock the _execute_with_algorithm method
        original_method = self.service._execute_with_algorithm
        
        async def mock_execute_with_algorithm(order, algorithm, algorithm_config):
            # Create a mock execution result
            return ExecutionReport(
                order_id="algo-1",
                client_order_id=order.client_order_id,
                instrument=order.instrument,
                status=OrderStatus.FILLED,
                direction=order.direction,
                order_type=order.order_type,
                quantity=order.quantity,
                filled_quantity=order.quantity,
                price=order.price,
                executed_price=order.price,
            )
            
        self.service._execute_with_algorithm = mock_execute_with_algorithm
        
        # Place the order with the VWAP algorithm
        report = self.service.place_order(
            self.order,
            algorithm=ExecutionAlgorithm.VWAP,
            algorithm_config={"duration_minutes": 30, "num_slices": 6}
        )
        
        # Check the execution report
        self.assertEqual(report.client_order_id, self.order.client_order_id)
        self.assertEqual(report.instrument, self.order.instrument)
        self.assertEqual(report.status, OrderStatus.FILLED)
        
        # Restore the original method
        self.service._execute_with_algorithm = original_method
    
    def test_implementation_shortfall_algorithm(self):
        """Test execution with the Implementation Shortfall algorithm."""
        # Mock the _execute_with_algorithm method
        original_method = self.service._execute_with_algorithm
        
        async def mock_execute_with_algorithm(order, algorithm, algorithm_config):
            # Create a mock execution result
            return ExecutionReport(
                order_id="algo-1",
                client_order_id=order.client_order_id,
                instrument=order.instrument,
                status=OrderStatus.FILLED,
                direction=order.direction,
                order_type=order.order_type,
                quantity=order.quantity,
                filled_quantity=order.quantity,
                price=order.price,
                executed_price=order.price,
            )
            
        self.service._execute_with_algorithm = mock_execute_with_algorithm
        
        # Place the order with the Implementation Shortfall algorithm
        report = self.service.place_order(
            self.order,
            algorithm=ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL,
            algorithm_config={"urgency": 0.7}
        )
        
        # Check the execution report
        self.assertEqual(report.client_order_id, self.order.client_order_id)
        self.assertEqual(report.instrument, self.order.instrument)
        self.assertEqual(report.status, OrderStatus.FILLED)
        
        # Restore the original method
        self.service._execute_with_algorithm = original_method
    
    def test_create_algorithm_instance(self):
        """Test the _create_algorithm_instance method."""
        # Test SOR algorithm
        algo = self.service._create_algorithm_instance(
            ExecutionAlgorithm.SOR,
            {"min_brokers": 2}
        )
        self.assertIsInstance(algo, SmartOrderRoutingAlgorithm)
        
        # Test TWAP algorithm
        algo = self.service._create_algorithm_instance(
            ExecutionAlgorithm.TWAP,
            {"duration_minutes": 30}
        )
        self.assertIsInstance(algo, TWAPAlgorithm)
        
        # Test VWAP algorithm
        algo = self.service._create_algorithm_instance(
            ExecutionAlgorithm.VWAP,
            {"duration_minutes": 30}
        )
        self.assertIsInstance(algo, VWAPAlgorithm)
        
        # Test Implementation Shortfall algorithm
        algo = self.service._create_algorithm_instance(
            ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL,
            {"urgency": 0.7}
        )
        self.assertIsInstance(algo, ImplementationShortfallAlgorithm)
    
    def test_algorithm_callbacks(self):
        """Test algorithm callbacks."""
        # Create a mock algorithm instance
        algo = MagicMock(spec=BaseExecutionAlgorithm)
        algo.algorithm_id = "algo-1"
        
        # Register callbacks
        callback_data = {}
        
        def callback(data):
            callback_data[data["event"]] = data
            
        self.service.register_callback("algorithm_started", callback)
        self.service.register_callback("algorithm_progress", callback)
        self.service.register_callback("algorithm_completed", callback)
        
        # Trigger callbacks
        self.service._trigger_callbacks("algorithm_started", {"event": "started", "algorithm_id": "algo-1"})
        self.service._trigger_callbacks("algorithm_progress", {"event": "progress", "algorithm_id": "algo-1", "progress": 50})
        self.service._trigger_callbacks("algorithm_completed", {"event": "completed", "algorithm_id": "algo-1"})
        
        # Check that callbacks were triggered
        self.assertEqual(callback_data["started"]["algorithm_id"], "algo-1")
        self.assertEqual(callback_data["progress"]["progress"], 50)
        self.assertEqual(callback_data["completed"]["algorithm_id"], "algo-1")


if __name__ == "__main__":
    unittest.main()
