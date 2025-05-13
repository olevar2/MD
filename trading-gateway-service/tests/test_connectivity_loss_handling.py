"""
Unit tests for connectivity loss handling in broker adapters.

This module contains tests for the reconnection logic, local order queuing,
position reconciliation, and heartbeat monitoring in broker adapters.
"""

import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from datetime import datetime, timedelta

from adapters.base_broker_adapter import (
    BaseBrokerAdapter, ConnectionState
)
from interfaces.broker_adapter_interface import (
    OrderRequest, OrderStatus, OrderDirection, OrderType, ExecutionReport
)


class TestBrokerAdapter(BaseBrokerAdapter):
    """Test implementation of BaseBrokerAdapter for testing."""

    def __init__(self, name="TestBroker", config=None):
    """
      init  .

    Args:
        name: Description of name
        config: Description of config

    """

        super().__init__(name, config or {})
        self.connect_broker_called = False
        self.connect_broker_return = True
        self.disconnect_broker_called = False
        self.disconnect_broker_return = True
        self.place_order_broker_called = False
        self.place_order_broker_return = None
        self.cancel_order_broker_called = False
        self.cancel_order_broker_return = None
        self.modify_order_broker_called = False
        self.modify_order_broker_return = None
        self.get_orders_broker_called = False
        self.get_orders_broker_return = []
        self.get_positions_broker_called = False
        self.get_positions_broker_return = []
        self.get_account_info_broker_called = False
        self.get_account_info_broker_return = {}
        self.get_broker_info_broker_called = False
        self.get_broker_info_broker_return = None
        self.check_connection_broker_called = False
        self.check_connection_broker_return = True

    def _connect_broker(self) -> bool:
    """
     connect broker.

    Returns:
        bool: Description of return value

    """

        self.connect_broker_called = True
        return self.connect_broker_return

    def _disconnect_broker(self) -> bool:
    """
     disconnect broker.

    Returns:
        bool: Description of return value

    """

        self.disconnect_broker_called = True
        return self.disconnect_broker_return

    def _place_order_broker(self, order: OrderRequest) -> ExecutionReport:
    """
     place order broker.

    Args:
        order: Description of order

    Returns:
        ExecutionReport: Description of return value

    """

        self.place_order_broker_called = True
        return self.place_order_broker_return or ExecutionReport(
            order_id="test-order-1",
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

    def _cancel_order_broker(self, order_id: str) -> ExecutionReport:
    """
     cancel order broker.

    Args:
        order_id: Description of order_id

    Returns:
        ExecutionReport: Description of return value

    """

        self.cancel_order_broker_called = True
        return self.cancel_order_broker_return or ExecutionReport(
            order_id=order_id,
            client_order_id="test-client-id",
            instrument="EURUSD",
            status=OrderStatus.CANCELLED,
        )

    def _modify_order_broker(self, order_id: str, modifications: dict) -> ExecutionReport:
    """
     modify order broker.

    Args:
        order_id: Description of order_id
        modifications: Description of modifications

    Returns:
        ExecutionReport: Description of return value

    """

        self.modify_order_broker_called = True
        return self.modify_order_broker_return or ExecutionReport(
            order_id=order_id,
            client_order_id="test-client-id",
            instrument="EURUSD",
            status=OrderStatus.ACCEPTED,
        )

    def _get_orders_broker(self) -> list:
    """
     get orders broker.

    Returns:
        list: Description of return value

    """

        self.get_orders_broker_called = True
        return self.get_orders_broker_return

    def _get_positions_broker(self) -> list:
    """
     get positions broker.

    Returns:
        list: Description of return value

    """

        self.get_positions_broker_called = True
        return self.get_positions_broker_return

    def _get_account_info_broker(self) -> dict:
    """
     get account info broker.

    Returns:
        dict: Description of return value

    """

        self.get_account_info_broker_called = True
        return self.get_account_info_broker_return

    def _get_broker_info_broker(self) -> dict:
    """
     get broker info broker.

    Returns:
        dict: Description of return value

    """

        self.get_broker_info_broker_called = True
        return self.get_broker_info_broker_return

    def _check_connection_broker(self) -> bool:
        self.check_connection_broker_called = True
        return self.check_connection_broker_return


class TestConnectivityLossHandling(unittest.TestCase):
    """Tests for connectivity loss handling in broker adapters."""

    def setUp(self):
        """Set up test fixtures."""
        self.adapter = TestBrokerAdapter()

    async def test_reconnection_logic(self):
        """Test reconnection logic with exponential backoff."""
        # Setup
        self.adapter._connection_state = ConnectionState.DISCONNECTED
        self.adapter._connection_attempts = 0
        self.adapter._max_reconnect_attempts = 3
        self.adapter._reconnect_delay_base = 0.1  # Short delay for testing
        self.adapter._reconnect_delay_max = 1.0

        # Mock connect method to fail first, then succeed
        original_connect = self.adapter.connect
        connect_calls = 0

        async def mock_connect(credentials):
            """
            Mock connect.

            Args:
                credentials: Description of credentials
            """
            nonlocal connect_calls
            connect_calls += 1
            if connect_calls == 1:
                return False
            return await original_connect(credentials)

        self.adapter.connect = mock_connect

        # Test
        result = await self.adapter.reconnect()

        # Verify
        self.assertTrue(result)
        self.assertEqual(self.adapter._connection_state, ConnectionState.CONNECTED)
        self.assertEqual(self.adapter._connection_attempts, 2)
        self.assertEqual(self.adapter._metrics["reconnect_attempts"], 2)
        self.assertEqual(self.adapter._metrics["successful_reconnects"], 1)
        self.assertEqual(self.adapter._metrics["failed_reconnects"], 1)

    async def test_reconnection_max_attempts(self):
        """Test reconnection logic with maximum attempts reached."""
        # Setup
        self.adapter._connection_state = ConnectionState.DISCONNECTED
        self.adapter._connection_attempts = 3
        self.adapter._max_reconnect_attempts = 3
        self.adapter.connect_broker_return = False

        # Test
        result = await self.adapter.reconnect()

        # Verify
        self.assertFalse(result)
        self.assertEqual(self.adapter._connection_state, ConnectionState.ERROR)
        self.assertEqual(self.adapter._connection_attempts, 4)
        self.assertEqual(self.adapter._metrics["reconnect_attempts"], 1)
        self.assertEqual(self.adapter._metrics["failed_reconnects"], 1)

    async def test_local_order_queuing(self):
        """Test local order queuing during disconnects."""
        # Setup
        self.adapter._connection_state = ConnectionState.DISCONNECTED
        self.adapter._enable_order_queuing = True

        order = OrderRequest(
            instrument="EURUSD",
            order_type=OrderType.MARKET,
            direction=OrderDirection.BUY,
            quantity=10000,
            price=1.1000,
            client_order_id="test-order-1"
        )

        # Test
        result = self.adapter.place_order(order)

        # Verify
        self.assertEqual(result.status, OrderStatus.PENDING)
        self.assertEqual(result.client_order_id, "test-order-1")
        self.assertEqual(len(self.adapter._order_queue), 1)
        self.assertEqual(self.adapter._metrics["orders_queued"], 1)

        # Now reconnect and verify queued orders are processed
        self.adapter._connection_state = ConnectionState.CONNECTED
        await self.adapter._process_queued_orders()

        # Verify
        self.assertEqual(len(self.adapter._order_queue), 0)
        self.assertTrue(self.adapter.place_order_broker_called)

    async def test_heartbeat_monitoring(self):
        """Test heartbeat monitoring."""
        # Setup
        self.adapter._connection_state = ConnectionState.CONNECTED
        self.adapter._heartbeat_interval = 0.1  # Short interval for testing
        self.adapter._last_heartbeat_time = datetime.now() - timedelta(seconds=1)

        # Mock check_connection method
        original_check_connection = self.adapter.check_connection
        check_connection_calls = 0

        async def mock_check_connection():
    """
    Mock check connection.

    """

            nonlocal check_connection_calls
            check_connection_calls += 1
            return await original_check_connection()

        self.adapter.check_connection = mock_check_connection

        # Start heartbeat monitoring
        self.adapter._start_heartbeat_monitoring()

        # Wait for heartbeat to trigger
        await asyncio.sleep(0.2)

        # Stop heartbeat monitoring
        self.adapter._stop_heartbeat_monitoring()

        # Verify
        self.assertTrue(check_connection_calls > 0)
        self.assertTrue(self.adapter.check_connection_broker_called)

    async def test_position_reconciliation(self):
        """Test position reconciliation after reconnect."""
        # Setup
        self.adapter._connection_state = ConnectionState.CONNECTED
        self.adapter._enable_position_reconciliation = True

        # Mock get_positions method
        self.adapter.get_positions_broker_return = [
            {"instrument": "EURUSD", "position_id": "1", "quantity": 10000},
            {"instrument": "GBPUSD", "position_id": "2", "quantity": -5000}
        ]

        # Mock the position callback
        position_updates = []

        async def mock_position_callback(position):
    """
    Mock position callback.

    Args:
        position: Description of position

    """

            position_updates.append(position)

        self.adapter._position_callback = mock_position_callback

        # Test
        await self.adapter._reconcile_positions()

        # Verify
        self.assertTrue(self.adapter.get_positions_broker_called)
        self.assertEqual(len(position_updates), 2)
        self.assertEqual(position_updates[0]["instrument"], "EURUSD")
        self.assertEqual(position_updates[1]["instrument"], "GBPUSD")


if __name__ == '__main__':
    unittest.main()
