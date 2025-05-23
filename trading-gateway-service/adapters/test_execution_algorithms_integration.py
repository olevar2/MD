"""
Integration tests for execution algorithms with broker adapters.

This module contains tests for the integration between execution algorithms
and broker adapters, ensuring they work together correctly.
"""
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from datetime import datetime
from interfaces.broker_adapter_interface import BrokerAdapterInterface, OrderRequest, OrderStatus, OrderDirection, OrderType, ExecutionReport
from trading_gateway_service.execution_algorithms import SmartOrderRoutingAlgorithm, TWAPAlgorithm, VWAPAlgorithm, ImplementationShortfallAlgorithm


from utils.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class MockBrokerAdapter(BrokerAdapterInterface):
    """Mock broker adapter for testing."""

    def __init__(self, name: str):
    """
      init  .
    
    Args:
        name: Description of name
    
    """

        self.name = name
        self.orders = []
        self.connected = True
        self.execution_reports = {}

    def is_connected(self) ->bool:
        return self.connected

    async def place_order(self, order: OrderRequest) ->ExecutionReport:
        """Mock place_order method."""
        self.orders.append(order)
        report = ExecutionReport(order_id=f'order-{len(self.orders)}',
            client_order_id=order.client_order_id, instrument=order.
            instrument, status=OrderStatus.FILLED, direction=order.
            direction, order_type=order.order_type, quantity=order.quantity,
            filled_quantity=order.quantity, price=order.price,
            executed_price=order.price)
        self.execution_reports[order.client_order_id] = report
        return report

    async def cancel_order(self, client_order_id: str, broker_order_id: str
        =None) ->ExecutionReport:
        """Mock cancel_order method."""
        order = next((o for o in self.orders if o.client_order_id ==
            client_order_id), None)
        if not order:
            return ExecutionReport(order_id='', client_order_id=
                client_order_id, instrument='', status=OrderStatus.REJECTED,
                rejection_reason='Order not found')
        report = ExecutionReport(order_id=broker_order_id or
            f'order-{self.orders.index(order) + 1}', client_order_id=
            client_order_id, instrument=order.instrument, status=
            OrderStatus.CANCELLED, direction=order.direction, order_type=
            order.order_type, quantity=order.quantity, filled_quantity=0,
            price=order.price, executed_price=None)
        self.execution_reports[client_order_id] = report
        return report

    async def modify_order(self, client_order_id: str, modifications: dict,
        broker_order_id: str=None) ->ExecutionReport:
        """Mock modify_order method."""
        order = next((o for o in self.orders if o.client_order_id ==
            client_order_id), None)
        if not order:
            return ExecutionReport(order_id='', client_order_id=
                client_order_id, instrument='', status=OrderStatus.REJECTED,
                rejection_reason='Order not found')
        for key, value in modifications.items():
            setattr(order, key, value)
        report = ExecutionReport(order_id=broker_order_id or
            f'order-{self.orders.index(order) + 1}', client_order_id=
            client_order_id, instrument=order.instrument, status=
            OrderStatus.ACCEPTED, direction=order.direction, order_type=
            order.order_type, quantity=order.quantity, filled_quantity=0,
            price=order.price, executed_price=None)
        self.execution_reports[client_order_id] = report
        return report

    @with_broker_api_resilience('get_order_status')
    async def get_order_status(self, client_order_id: str, broker_order_id:
        str=None) ->ExecutionReport:
        """Mock get_order_status method."""
        if client_order_id in self.execution_reports:
            return self.execution_reports[client_order_id]
        order = next((o for o in self.orders if o.client_order_id ==
            client_order_id), None)
        if not order:
            return ExecutionReport(order_id='', client_order_id=
                client_order_id, instrument='', status=OrderStatus.REJECTED,
                rejection_reason='Order not found')
        report = ExecutionReport(order_id=broker_order_id or
            f'order-{self.orders.index(order) + 1}', client_order_id=
            client_order_id, instrument=order.instrument, status=
            OrderStatus.PENDING, direction=order.direction, order_type=
            order.order_type, quantity=order.quantity, filled_quantity=0,
            price=order.price, executed_price=None)
        self.execution_reports[client_order_id] = report
        return report

    @with_broker_api_resilience('get_positions')
    async def get_positions(self) ->list:
        """Mock get_positions method."""
        return []

    @with_broker_api_resilience('get_account_info')
    async def get_account_info(self) ->dict:
        """Mock get_account_info method."""
        return {'balance': 10000.0, 'equity': 10000.0, 'margin_used': 0.0,
            'margin_available': 10000.0, 'currency': 'USD'}

    @with_broker_api_resilience('get_broker_info')
    async def get_broker_info(self) ->dict:
        """Mock get_broker_info method."""
        return {'name': self.name, 'supported_order_types': [OrderType.
            MARKET, OrderType.LIMIT, OrderType.STOP],
            'supported_instruments': ['EURUSD', 'GBPUSD', 'USDJPY'],
            'min_order_size': 1000, 'max_order_size': 10000000,
            'price_precision': 5, 'size_precision': 2}

    async def connect(self, credentials: dict) ->bool:
        """Mock connect method."""
        self.connected = True
        return True

    async def disconnect(self) ->bool:
        """Mock disconnect method."""
        self.connected = False
        return True


class TestExecutionAlgorithmsIntegration(unittest.TestCase):
    """Integration tests for execution algorithms with broker adapters."""

    def setUp(self):
        """Set up test fixtures."""
        self.broker_adapters = {'broker1': MockBrokerAdapter('broker1'),
            'broker2': MockBrokerAdapter('broker2'), 'broker3':
            MockBrokerAdapter('broker3')}
        self.order = OrderRequest(instrument='EURUSD', order_type=OrderType
            .MARKET, direction=OrderDirection.BUY, quantity=10000, price=
            1.1, client_order_id='test-order-1')

    async def test_smart_order_routing_algorithm(self):
        """Test Smart Order Routing algorithm with broker adapters."""
        algorithm = SmartOrderRoutingAlgorithm(broker_adapters=self.
            broker_adapters, config={'price_weight': 0.4, 'speed_weight': 
            0.3, 'liquidity_weight': 0.2, 'reliability_weight': 0.1,
            'min_brokers': 1, 'max_brokers': 3, 'min_order_size': 1000})
        original_score_brokers = algorithm._score_brokers

        async def mock_score_brokers(instrument):
    """
    Mock score brokers.
    
    Args:
        instrument: Description of instrument
    
    """

            return {'broker1': 0.9, 'broker2': 0.7, 'broker3': 0.5}
        algorithm._score_brokers = mock_score_brokers
        result = await algorithm.execute(self.order)
        self.assertEqual(result.status, 'COMPLETED')
        self.assertEqual(result.original_order_id, 'test-order-1')
        self.assertEqual(len(result.execution_reports), 1)
        self.assertEqual(result.total_filled_quantity, 10000)
        self.assertEqual(len(self.broker_adapters['broker1'].orders), 1)
        self.assertEqual(len(self.broker_adapters['broker2'].orders), 0)
        self.assertEqual(len(self.broker_adapters['broker3'].orders), 0)

    async def test_twap_algorithm(self):
        """Test TWAP algorithm with broker adapters."""
        algorithm = TWAPAlgorithm(broker_adapters=self.broker_adapters,
            config={'default_broker': 'broker1', 'duration_minutes': 0.1,
            'num_slices': 2, 'min_slice_size': 1000})
        result = await algorithm.execute(self.order)
        self.assertEqual(result.status, 'COMPLETED')
        self.assertEqual(result.original_order_id, 'test-order-1')
        self.assertEqual(len(result.execution_reports), 2)
        self.assertEqual(result.total_filled_quantity, 10000)
        self.assertEqual(len(self.broker_adapters['broker1'].orders), 2)
        self.assertEqual(self.broker_adapters['broker1'].orders[0].quantity,
            5000)
        self.assertEqual(self.broker_adapters['broker1'].orders[1].quantity,
            5000)

    async def test_vwap_algorithm(self):
        """Test VWAP algorithm with broker adapters."""
        algorithm = VWAPAlgorithm(broker_adapters=self.broker_adapters,
            config={'default_broker': 'broker1', 'duration_minutes': 0.1,
            'num_slices': 2, 'min_slice_size': 1000,
            'volume_profile_source': 'default'})
        original_get_volume_profile = algorithm._get_volume_profile

        def mock_get_volume_profile(instrument, start_time, end_time):
    """
    Mock get volume profile.
    
    Args:
        instrument: Description of instrument
        start_time: Description of start_time
        end_time: Description of end_time
    
    """

            return [0.6, 0.4]
        algorithm._get_volume_profile = mock_get_volume_profile
        result = await algorithm.execute(self.order)
        self.assertEqual(result.status, 'COMPLETED')
        self.assertEqual(result.original_order_id, 'test-order-1')
        self.assertEqual(len(result.execution_reports), 2)
        self.assertEqual(result.total_filled_quantity, 10000)
        self.assertEqual(len(self.broker_adapters['broker1'].orders), 2)
        self.assertEqual(self.broker_adapters['broker1'].orders[0].quantity,
            6000)
        self.assertEqual(self.broker_adapters['broker1'].orders[1].quantity,
            4000)

    async def test_implementation_shortfall_algorithm(self):
        """Test Implementation Shortfall algorithm with broker adapters."""
        algorithm = ImplementationShortfallAlgorithm(broker_adapters=self.
            broker_adapters, config={'default_broker': 'broker1',
            'max_duration_minutes': 0.1, 'min_duration_minutes': 0.05,
            'min_slice_size': 1000, 'urgency': 0.5})
        original_get_market_conditions = algorithm._get_market_conditions

        async def mock_get_market_conditions(instrument):
    """
    Mock get market conditions.
    
    Args:
        instrument: Description of instrument
    
    """

            return {'price': 1.1, 'spread': 0.0001, 'volatility': 0.001,
                'avg_daily_volume': 1000000, 'market_regime': 'normal'}
        algorithm._get_market_conditions = mock_get_market_conditions
        result = await algorithm.execute(self.order)
        self.assertEqual(result.status, 'COMPLETED')
        self.assertEqual(result.original_order_id, 'test-order-1')
        self.assertTrue(len(result.execution_reports) > 0)
        self.assertEqual(result.total_filled_quantity, 10000)
        self.assertTrue(len(self.broker_adapters['broker1'].orders) > 0)
        self.assertEqual(sum(order.quantity for order in self.
            broker_adapters['broker1'].orders), 10000)


if __name__ == '__main__':
    unittest.main()
