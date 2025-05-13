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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
    '../..')))
from trading_gateway_service.interfaces.broker_adapter_interface import OrderRequest, OrderStatus, OrderDirection, OrderType, ExecutionReport
from trading_gateway_service.services.order_execution_service import OrderExecutionService
from trading_gateway_service.services.execution import ExecutionMode, ExecutionAlgorithm
from tests.services.mock_broker_adapter_interface import MockBrokerAdapterInterface


from trading_gateway_service.resilience.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class MockBrokerAdapter(MockBrokerAdapterInterface):
    """Mock broker adapter for testing."""

    def __init__(self, name: str):
        """Initialize the mock broker adapter."""
        self.name = name
        self.connected = False
        self.orders = []

    def is_connected(self) ->bool:
        """Check if the broker is connected."""
        return self.connected

    def connect(self, credentials=None) ->bool:
        """Connect to the broker."""
        self.connected = True
        return True

    def disconnect(self) ->bool:
        """Disconnect from the broker."""
        self.connected = False
        return True

    def place_order(self, order_request):
        """Place an order with the broker."""
        self.orders.append(order_request)
        return ExecutionReport(order_id=f'{self.name}-{len(self.orders)}',
            client_order_id=order_request.client_order_id, instrument=
            order_request.instrument, status=OrderStatus.OPEN, direction=
            order_request.direction, order_type=order_request.order_type,
            quantity=order_request.quantity, filled_quantity=0.0, price=
            order_request.price)

    def cancel_order(self, order_id):
        """Cancel an order with the broker."""
        return ExecutionReport(order_id=order_id, client_order_id='test',
            instrument='EURUSD', status=OrderStatus.CANCELLED, direction=
            OrderDirection.BUY, order_type=OrderType.MARKET, quantity=10000,
            filled_quantity=0.0, price=1.1)

    def modify_order(self, order_id, modifications):
        """Modify an order with the broker."""
        return ExecutionReport(order_id=order_id, client_order_id='test',
            instrument='EURUSD', status=OrderStatus.OPEN, direction=
            OrderDirection.BUY, order_type=OrderType.MARKET, quantity=
            modifications.get('quantity', 10000), filled_quantity=0.0,
            price=modifications.get('price', 1.1))

    @with_broker_api_resilience('get_orders')
    def get_orders(self):
        """Get all orders from the broker."""
        return self.orders

    @with_broker_api_resilience('get_positions')
    def get_positions(self):
        """Get all positions from the broker."""
        return []

    @with_broker_api_resilience('get_account_info')
    def get_account_info(self):
        """Get account information from the broker."""
        return {'account_id': f'{self.name}-account', 'balance': 100000.0,
            'currency': 'USD'}

    @with_broker_api_resilience('get_broker_info')
    def get_broker_info(self):
        """Get broker information."""
        return {'name': self.name, 'description':
            f'Mock broker adapter for {self.name}', 'version': '1.0.0'}

    @with_market_data_resilience('get_market_data')
    def get_market_data(self, instrument, data_type=None):
        """Get market data from the broker."""
        return {'instrument': instrument, 'bid': 1.1, 'ask': 1.1001,
            'timestamp': datetime.utcnow()}


class TestRefactoredOrderExecutionService(unittest.TestCase):
    """Tests for the refactored OrderExecutionService."""

    def setUp(self):
        """Set up test fixtures."""
        self.service = OrderExecutionService(mode=ExecutionMode.SIMULATED)
        self.broker1 = MockBrokerAdapter('broker1')
        self.broker2 = MockBrokerAdapter('broker2')
        self.broker3 = MockBrokerAdapter('broker3')
        self.service.register_broker_adapter('broker1', self.broker1,
            default=True)
        self.service.register_broker_adapter('broker2', self.broker2)
        self.service.register_broker_adapter('broker3', self.broker3)
        self.market_order = OrderRequest(instrument='EURUSD', order_type=
            OrderType.MARKET, direction=OrderDirection.BUY, quantity=10000,
            price=1.1, client_order_id='test-market-order')
        self.limit_order = OrderRequest(instrument='EURUSD', order_type=
            OrderType.LIMIT, direction=OrderDirection.BUY, quantity=10000,
            price=1.1, client_order_id='test-limit-order')
        self.stop_order = OrderRequest(instrument='EURUSD', order_type=
            OrderType.STOP, direction=OrderDirection.BUY, quantity=10000,
            price=1.1, client_order_id='test-stop-order')

    def test_register_broker_adapter(self):
        """Test registering a broker adapter."""
        broker4 = MockBrokerAdapter('broker4')
        self.service.register_broker_adapter('broker4', broker4)
        self.assertIn('broker4', self.service.broker_adapters)
        self.assertEqual(self.service.default_broker, 'broker1')
        broker5 = MockBrokerAdapter('broker5')
        self.service.register_broker_adapter('broker5', broker5, default=True)
        self.assertIn('broker5', self.service.broker_adapters)
        self.assertEqual(self.service.default_broker, 'broker5')

    def test_set_default_broker(self):
        """Test setting the default broker."""
        self.service.default_broker = 'broker2'
        self.assertEqual(self.service.default_broker, 'broker2')

    def test_connect(self):
        """Test connecting to brokers."""
        result = self.service.connect()
        self.assertTrue(result)
        self.assertTrue(self.broker1.is_connected())
        self.assertTrue(self.broker2.is_connected())
        self.assertTrue(self.broker3.is_connected())
        self.broker1.connected = False
        self.broker2.connected = False
        self.broker3.connected = False
        result = self.service.connect('broker1')
        self.assertTrue(result)
        self.assertTrue(self.broker1.is_connected())
        self.assertFalse(self.broker2.is_connected())
        self.assertFalse(self.broker3.is_connected())
        result = self.service.connect('non-existent')
        self.assertFalse(result)

    def test_disconnect(self):
        """Test disconnecting from brokers."""
        self.broker1.connected = True
        self.broker2.connected = True
        self.broker3.connected = True
        result = self.service.disconnect()
        self.assertTrue(result)
        self.assertFalse(self.broker1.is_connected())
        self.assertFalse(self.broker2.is_connected())
        self.assertFalse(self.broker3.is_connected())
        self.broker1.connected = True
        self.broker2.connected = True
        self.broker3.connected = True
        result = self.service.disconnect('broker1')
        self.assertTrue(result)
        self.assertFalse(self.broker1.is_connected())
        self.assertTrue(self.broker2.is_connected())
        self.assertTrue(self.broker3.is_connected())
        result = self.service.disconnect('non-existent')
        self.assertFalse(result)

    def test_place_market_order(self):
        """Test placing a market order."""
        report = self.service.place_order(self.market_order)
        self.assertEqual(report.client_order_id, self.market_order.
            client_order_id)
        self.assertEqual(report.instrument, self.market_order.instrument)
        self.assertEqual(report.status, OrderStatus.OPEN)
        self.assertEqual(len(self.broker1.orders), 1)
        self.assertEqual(self.broker1.orders[0].client_order_id, self.
            market_order.client_order_id)
        report = self.service.place_order(self.market_order, broker_name=
            'broker2')
        self.assertEqual(report.client_order_id, self.market_order.
            client_order_id)
        self.assertEqual(report.instrument, self.market_order.instrument)
        self.assertEqual(report.status, OrderStatus.OPEN)
        self.assertEqual(len(self.broker2.orders), 1)
        self.assertEqual(self.broker2.orders[0].client_order_id, self.
            market_order.client_order_id)
        report = self.service.place_order(self.market_order, broker_name=
            'non-existent')
        self.assertEqual(report.status, OrderStatus.REJECTED)

    def test_place_limit_order(self):
        """Test placing a limit order."""
        report = self.service.place_order(self.limit_order)
        self.assertEqual(report.client_order_id, self.limit_order.
            client_order_id)
        self.assertEqual(report.instrument, self.limit_order.instrument)
        self.assertEqual(report.status, OrderStatus.OPEN)
        self.assertEqual(len(self.broker1.orders), 1)
        self.assertEqual(self.broker1.orders[0].client_order_id, self.
            limit_order.client_order_id)

    def test_place_stop_order(self):
        """Test placing a stop order."""
        report = self.service.place_order(self.stop_order)
        self.assertEqual(report.client_order_id, self.stop_order.
            client_order_id)
        self.assertEqual(report.instrument, self.stop_order.instrument)
        self.assertEqual(report.status, OrderStatus.OPEN)
        self.assertEqual(len(self.broker1.orders), 1)
        self.assertEqual(self.broker1.orders[0].client_order_id, self.
            stop_order.client_order_id)

    def test_cancel_order(self):
        """Test cancelling an order."""
        report = self.service.place_order(self.market_order)
        order_id = report.order_id
        cancel_report = self.service.cancel_order(order_id)
        self.assertEqual(cancel_report.order_id, order_id)
        self.assertEqual(cancel_report.status, OrderStatus.CANCELLED)
        cancel_report = self.service.cancel_order('non-existent')
        self.assertEqual(cancel_report.status, OrderStatus.REJECTED)

    def test_modify_order(self):
        """Test modifying an order."""
        report = self.service.place_order(self.market_order)
        order_id = report.order_id
        modifications = {'quantity': 20000, 'price': 1.2}
        modify_report = self.service.modify_order(order_id, modifications)
        self.assertEqual(modify_report.order_id, order_id)
        self.assertEqual(modify_report.quantity, 20000)
        self.assertEqual(modify_report.price, 1.2)
        modify_report = self.service.modify_order('non-existent', modifications
            )
        self.assertEqual(modify_report.status, OrderStatus.REJECTED)

    def test_get_orders(self):
        """Test getting orders."""
        self.service.place_order(self.market_order)
        self.service.place_order(self.limit_order)
        self.service.place_order(self.stop_order)
        orders = self.service.get_orders()
        self.assertEqual(len(orders), 3)
        orders = self.service.get_orders(instrument='EURUSD')
        self.assertEqual(len(orders), 3)
        orders = self.service.get_orders(status=OrderStatus.OPEN)
        self.assertEqual(len(orders), 3)
        orders = self.service.get_orders(instrument='EURUSD', status=
            OrderStatus.OPEN)
        self.assertEqual(len(orders), 3)
        orders = self.service.get_orders(instrument='GBPUSD')
        self.assertEqual(len(orders), 0)
        orders = self.service.get_orders(status=OrderStatus.FILLED)
        self.assertEqual(len(orders), 0)

    def test_get_order(self):
        """Test getting a specific order."""
        report = self.service.place_order(self.market_order)
        order_id = report.order_id
        order = self.service.get_order(order_id)
        self.assertIsNotNone(order)
        self.assertEqual(order['execution_report'].order_id, order_id)
        order = self.service.get_order('non-existent')
        self.assertIsNone(order)

    def test_update_execution_status(self):
        """Test updating the execution status of an order."""
        report = self.service.place_order(self.market_order)
        order_id = report.order_id
        status_update = {'status': OrderStatus.FILLED, 'filled_quantity': 
            10000, 'executed_price': 1.1}
        result = self.service.update_execution_status(order_id, status_update)
        self.assertTrue(result)
        order = self.service.get_order(order_id)
        self.assertEqual(order['execution_report'].status, OrderStatus.FILLED)
        self.assertEqual(order['execution_report'].filled_quantity, 10000)
        self.assertEqual(order['execution_report'].executed_price, 1.1)
        result = self.service.update_execution_status('non-existent',
            status_update)
        self.assertFalse(result)

    def test_algorithm_execution(self):
        """Test executing an order with an algorithm."""
        self.service.algorithm_execution_service = MagicMock()
        mock_report = ExecutionReport(order_id='algo-1', client_order_id=
            self.market_order.client_order_id, instrument=self.market_order
            .instrument, status=OrderStatus.FILLED, direction=self.
            market_order.direction, order_type=self.market_order.order_type,
            quantity=self.market_order.quantity, filled_quantity=self.
            market_order.quantity, price=self.market_order.price,
            executed_price=self.market_order.price)
        (self.service.algorithm_execution_service.place_order.return_value
            ) = mock_report
        report = self.service.place_order(self.market_order, algorithm=
            ExecutionAlgorithm.SOR, algorithm_config={'min_brokers': 2})
        self.service.algorithm_execution_service.place_order.assert_called_once(
            )
        self.assertEqual(report.client_order_id, self.market_order.
            client_order_id)
        self.assertEqual(report.instrument, self.market_order.instrument)
        self.assertEqual(report.status, OrderStatus.FILLED)

    def test_get_algorithm_status(self):
        """Test getting the status of an algorithm."""
        self.service.algorithm_execution_service = MagicMock()
        (self.service.algorithm_execution_service.get_algorithm_status.
            return_value) = ({'algorithm_id': 'algo-1', 'status': 'running',
            'progress': 0.5, 'filled_quantity': 5000, 'remaining_quantity':
            5000})
        status = self.service.get_algorithm_status('algo-1')
        self.service.algorithm_execution_service.get_algorithm_status.assert_called_once_with(
            'algo-1')
        self.assertIsNotNone(status)
        self.assertEqual(status['status'], 'running')
        self.assertEqual(status['progress'], 0.5)
        (self.service.algorithm_execution_service.get_algorithm_status.
            return_value) = None
        status = self.service.get_algorithm_status('non-existent')
        self.assertIsNone(status)

    def test_get_active_algorithms(self):
        """Test getting active algorithms."""
        self.service.algorithm_execution_service = MagicMock()
        (self.service.algorithm_execution_service.get_active_algorithms.
            return_value) = ['algo-1', 'algo-2']
        algorithms = self.service.get_active_algorithms()
        self.service.algorithm_execution_service.get_active_algorithms.assert_called_once(
            )
        self.assertEqual(len(algorithms), 2)
        self.assertIn('algo-1', algorithms)
        self.assertIn('algo-2', algorithms)
        (self.service.algorithm_execution_service.get_active_algorithms.
            return_value) = []
        algorithms = self.service.get_active_algorithms()
        self.assertEqual(len(algorithms), 0)


if __name__ == '__main__':
    print('Running tests for refactored OrderExecutionService...')
    result = unittest.main(exit=False)
    print(f'Tests completed with result: {result.result}')
