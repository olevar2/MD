"""
Unit tests for execution algorithms.

This module contains tests for the execution algorithms implemented in the
trading_gateway_service.execution_algorithms package.
"""
import asyncio
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any
from interfaces.broker_adapter_interface import OrderRequest, OrderStatus, OrderDirection, OrderType, ExecutionReport
from trading_gateway_service.execution_algorithms import SmartOrderRoutingAlgorithm, TWAPAlgorithm, VWAPAlgorithm, ImplementationShortfallAlgorithm


from utils.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class MockBrokerAdapter:
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

    def is_connected(self) ->bool:
        return self.connected

    async def place_order(self, order: OrderRequest) ->ExecutionReport:
        """Mock place_order method."""
        self.orders.append(order)
        return ExecutionReport(order_id=f'order-{len(self.orders)}',
            client_order_id=order.client_order_id, instrument=order.
            instrument, status=OrderStatus.FILLED, direction=order.
            direction, order_type=order.order_type, quantity=order.quantity,
            filled_quantity=order.quantity, price=order.price,
            executed_price=order.price)

    @with_market_data_resilience('get_quote')
    async def get_quote(self, instrument: str) ->Dict[str, Any]:
        """Mock get_quote method."""
        return {'bid': 1.1, 'ask': 1.1001, 'spread': 0.0001, 'timestamp':
            datetime.utcnow().isoformat(), 'liquidity': {'bid': 1000000,
            'ask': 1000000}}


class TestSmartOrderRoutingAlgorithm(unittest.TestCase):
    """Tests for the Smart Order Routing algorithm."""

    def setUp(self):
        """Set up test fixtures."""
        self.broker_adapters = {'broker1': MockBrokerAdapter('broker1'),
            'broker2': MockBrokerAdapter('broker2'), 'broker3':
            MockBrokerAdapter('broker3')}
        self.algorithm = SmartOrderRoutingAlgorithm(broker_adapters=self.
            broker_adapters, config={'price_weight': 0.4, 'speed_weight': 
            0.3, 'liquidity_weight': 0.2, 'reliability_weight': 0.1,
            'min_brokers': 1, 'max_brokers': 3, 'min_order_size': 1000})
        self.order = OrderRequest(instrument='EURUSD', order_type=OrderType
            .MARKET, direction=OrderDirection.BUY, quantity=10000, price=
            1.1, client_order_id='test-order-1')

    async def test_execute(self):
        """Test the execute method."""
        result = await self.algorithm.execute(self.order)
        self.assertEqual(result.original_order_id, self.order.client_order_id)
        self.assertEqual(result.status, 'COMPLETED')
        self.assertTrue(len(result.execution_reports) > 0)
        self.assertEqual(result.total_filled_quantity, self.order.quantity)

    async def test_cancel(self):
        """Test the cancel method."""
        execution_task = asyncio.create_task(self.algorithm.execute(self.order)
            )
        await asyncio.sleep(0.1)
        cancelled = await self.algorithm.cancel()
        self.assertTrue(cancelled)
        result = await execution_task
        self.assertEqual(result.status, 'CANCELLED')

    def test_score_brokers(self):
        """Test the _score_brokers method."""

        async def test_coro():
            scores = await self.algorithm._score_brokers(self.order)
            return scores
        scores = asyncio.run(test_coro())
        self.assertEqual(len(scores), len(self.broker_adapters))
        for i in range(1, len(scores)):
            self.assertTrue(scores[i - 1].total_score >= scores[i].total_score)


class TestTWAPAlgorithm(unittest.TestCase):
    """Tests for the TWAP algorithm."""

    def setUp(self):
        """Set up test fixtures."""
        self.broker_adapters = {'broker1': MockBrokerAdapter('broker1')}
        self.algorithm = TWAPAlgorithm(broker_adapters=self.broker_adapters,
            config={'default_broker': 'broker1', 'duration_minutes': 0.1,
            'num_slices': 2, 'min_slice_size': 1000})
        self.order = OrderRequest(instrument='EURUSD', order_type=OrderType
            .MARKET, direction=OrderDirection.BUY, quantity=10000, price=
            1.1, client_order_id='test-order-1')

    async def test_execute(self):
        """Test the execute method."""
        result = await self.algorithm.execute(self.order)
        self.assertEqual(result.original_order_id, self.order.client_order_id)
        self.assertEqual(result.status, 'COMPLETED')
        self.assertEqual(len(result.execution_reports), 2)
        self.assertEqual(result.total_filled_quantity, self.order.quantity)

    async def test_cancel(self):
        """Test the cancel method."""
        execution_task = asyncio.create_task(self.algorithm.execute(self.order)
            )
        await asyncio.sleep(0.1)
        cancelled = await self.algorithm.cancel()
        self.assertTrue(cancelled)
        result = await execution_task
        self.assertEqual(result.status, 'CANCELLED')


class TestVWAPAlgorithm(unittest.TestCase):
    """Tests for the VWAP algorithm."""

    def setUp(self):
        """Set up test fixtures."""
        self.broker_adapters = {'broker1': MockBrokerAdapter('broker1')}
        self.algorithm = VWAPAlgorithm(broker_adapters=self.broker_adapters,
            config={'default_broker': 'broker1', 'duration_minutes': 0.1,
            'num_slices': 2, 'min_slice_size': 1000,
            'volume_profile_source': 'default'})
        self.order = OrderRequest(instrument='EURUSD', order_type=OrderType
            .MARKET, direction=OrderDirection.BUY, quantity=10000, price=
            1.1, client_order_id='test-order-1')

    async def test_execute(self):
        """Test the execute method."""
        result = await self.algorithm.execute(self.order)
        self.assertEqual(result.original_order_id, self.order.client_order_id)
        self.assertEqual(result.status, 'COMPLETED')
        self.assertEqual(len(result.execution_reports), 2)
        self.assertEqual(result.total_filled_quantity, self.order.quantity)

    async def test_cancel(self):
        """Test the cancel method."""
        execution_task = asyncio.create_task(self.algorithm.execute(self.order)
            )
        await asyncio.sleep(0.1)
        cancelled = await self.algorithm.cancel()
        self.assertTrue(cancelled)
        result = await execution_task
        self.assertEqual(result.status, 'CANCELLED')


class TestImplementationShortfallAlgorithm(unittest.TestCase):
    """Tests for the Implementation Shortfall algorithm."""

    def setUp(self):
        """Set up test fixtures."""
        self.broker_adapters = {'broker1': MockBrokerAdapter('broker1')}
        self.algorithm = ImplementationShortfallAlgorithm(broker_adapters=
            self.broker_adapters, config={'default_broker': 'broker1',
            'max_duration_minutes': 0.1, 'min_duration_minutes': 0.05,
            'min_slice_size': 1000, 'urgency': 0.5})
        self.order = OrderRequest(instrument='EURUSD', order_type=OrderType
            .MARKET, direction=OrderDirection.BUY, quantity=10000, price=
            1.1, client_order_id='test-order-1')
        self.algorithm._get_market_conditions = MagicMock(return_value={
            'price': 1.1, 'spread': 0.0001, 'volatility': 0.001,
            'avg_daily_volume': 1000000, 'market_regime': 'normal'})

    async def test_execute(self):
        """Test the execute method."""
        result = await self.algorithm.execute(self.order)
        self.assertEqual(result.original_order_id, self.order.client_order_id)
        self.assertEqual(result.status, 'COMPLETED')
        self.assertTrue(len(result.execution_reports) > 0)
        self.assertEqual(result.total_filled_quantity, self.order.quantity)

    async def test_cancel(self):
        """Test the cancel method."""
        execution_task = asyncio.create_task(self.algorithm.execute(self.order)
            )
        await asyncio.sleep(0.1)
        cancelled = await self.algorithm.cancel()
        self.assertTrue(cancelled)
        result = await execution_task
        self.assertEqual(result.status, 'CANCELLED')


if __name__ == '__main__':
    unittest.main()
