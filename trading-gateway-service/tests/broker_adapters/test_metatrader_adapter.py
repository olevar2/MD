"""
Unit tests for the MetaTrader adapter.

This module contains tests for the MetaTraderAdapter class, which provides
connectivity to MetaTrader 4/5 platforms.
"""

import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from datetime import datetime

from trading_gateway_service.broker_adapters.metatrader_adapter import MetaTraderAdapter
from trading_gateway_service.interfaces.broker_adapter import (
    OrderRequest, OrderStatus, OrderDirection, OrderType
)


class TestMetaTraderAdapter(unittest.TestCase):
    """Tests for the MetaTrader adapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the MetaTrader5 library
        self.mt5_patcher = patch('trading_gateway_service.broker_adapters.metatrader_adapter.mt5')
        self.mock_mt5 = self.mt5_patcher.start()
        
        # Configure mock responses
        self.mock_mt5.initialize.return_value = True
        self.mock_mt5.account_info.return_value = MagicMock(
            login=12345,
            server="MetaQuotes-Demo",
            balance=10000.0,
            equity=10050.0,
            margin=100.0,
            margin_free=9950.0,
            margin_level=100.5,
            currency="USD"
        )
        self.mock_mt5.terminal_info.return_value = MagicMock(connected=True)
        
        # Create adapter with test configuration
        self.config = {
            'path': 'C:\\Program Files\\MetaTrader 5\\terminal64.exe',
            'login': '12345',
            'password': 'password',
            'server': 'MetaQuotes-Demo',
            'connect_timeout_ms': 5000,
            'magic_number': 12345,
            'slippage_deviation': 10,
            'default_filling_type': 'FOK',
            'default_time_type': 'GTC',
            'polling_interval_seconds': 1.0
        }
        self.adapter = MetaTraderAdapter(self.config)
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.mt5_patcher.stop()
    
    async def test_connect(self):
        """Test connecting to MetaTrader."""
        # Test successful connection
        result = await self.adapter.connect()
        self.assertTrue(result)
        self.assertTrue(self.adapter.is_connected)
        self.mock_mt5.initialize.assert_called_once()
        self.mock_mt5.account_info.assert_called_once()
        
        # Test connection failure
        self.mock_mt5.initialize.reset_mock()
        self.mock_mt5.initialize.return_value = False
        self.mock_mt5.last_error.return_value = (10, "Failed to connect")
        
        self.adapter._connected = False
        result = await self.adapter.connect()
        self.assertFalse(result)
        self.assertFalse(self.adapter.is_connected)
    
    async def test_disconnect(self):
        """Test disconnecting from MetaTrader."""
        # Setup
        self.adapter._connected = True
        self.adapter._polling_task = None
        
        # Test
        result = await self.adapter.disconnect()
        self.assertTrue(result)
        self.assertFalse(self.adapter.is_connected)
        self.mock_mt5.shutdown.assert_called_once()
    
    async def test_place_order(self):
        """Test placing an order."""
        # Setup
        self.adapter._connected = True
        order_request = OrderRequest(
            instrument="EURUSD",
            order_type=OrderType.MARKET,
            direction=OrderDirection.BUY,
            quantity=0.1,
            price=1.1000,
            client_order_id="test-order-1"
        )
        
        # Mock order_send response
        self.mock_mt5.order_send.return_value = MagicMock(
            retcode=10009,  # Success code
            order=12345,
            volume=0.1,
            price=1.1000,
            comment="Filled",
            request=MagicMock(
                action=1,
                symbol="EURUSD",
                volume=0.1,
                type=0,
                price=1.1000
            )
        )
        
        # Mock order status check
        self.mock_mt5.orders_get.return_value = None
        self.mock_mt5.history_orders_get.return_value = [MagicMock(
            ticket=12345,
            symbol="EURUSD",
            type=0,  # BUY
            state=5,  # FILLED
            volume_initial=0.1,
            volume_current=0.1,
            price_open=1.1000,
            price_current=1.1000,
            sl=0.0,
            tp=0.0,
            time_setup=datetime.now().timestamp(),
            time_done=datetime.now().timestamp(),
            comment="test-order-1"
        )]
        
        # Test
        result = await self.adapter.place_order(order_request)
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertEqual(result.broker_order_id, "12345")
        self.assertEqual(result.client_order_id, "test-order-1")
        self.assertEqual(result.instrument, "EURUSD")
        self.assertEqual(result.filled_quantity, 0.1)
        self.mock_mt5.order_send.assert_called_once()
    
    async def test_cancel_order(self):
        """Test cancelling an order."""
        # Setup
        self.adapter._connected = True
        client_order_id = "test-order-1"
        broker_order_id = "12345"
        
        # Mock order_send response for cancellation
        self.mock_mt5.order_send.return_value = MagicMock(
            retcode=10009,  # Success code
            order=12345
        )
        
        # Mock order status check after cancellation
        self.mock_mt5.orders_get.return_value = None
        self.mock_mt5.history_orders_get.return_value = [MagicMock(
            ticket=12345,
            symbol="EURUSD",
            type=0,  # BUY
            state=4,  # CANCELLED
            volume_initial=0.1,
            volume_current=0.0,
            price_open=1.1000,
            price_current=1.1000,
            sl=0.0,
            tp=0.0,
            time_setup=datetime.now().timestamp(),
            time_done=datetime.now().timestamp(),
            comment="test-order-1"
        )]
        
        # Test
        result = await self.adapter.cancel_order(client_order_id, broker_order_id)
        self.assertEqual(result.status, OrderStatus.CANCELLED)
        self.assertEqual(result.broker_order_id, "12345")
        self.assertEqual(result.client_order_id, "test-order-1")
        self.mock_mt5.order_send.assert_called_once()
    
    async def test_get_positions(self):
        """Test getting positions."""
        # Setup
        self.adapter._connected = True
        
        # Mock positions_get response
        self.mock_mt5.positions_get.return_value = [MagicMock(
            ticket=12345,
            symbol="EURUSD",
            type=0,  # BUY
            volume=0.1,
            price_open=1.1000,
            price_current=1.1050,
            sl=1.0900,
            tp=1.1200,
            profit=5.0,
            margin=10.0,
            time=datetime.now().timestamp(),
            time_update=datetime.now().timestamp()
        )]
        
        # Test
        positions = await self.adapter.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].instrument, "EURUSD")
        self.assertEqual(positions[0].position_id, "12345")
        self.assertEqual(positions[0].quantity, 0.1)
        self.assertEqual(positions[0].average_price, 1.1000)
        self.assertEqual(positions[0].unrealized_pl, 5.0)
        self.mock_mt5.positions_get.assert_called_once()
    
    async def test_get_account_info(self):
        """Test getting account information."""
        # Setup
        self.adapter._connected = True
        
        # Test
        account_info = await self.adapter.get_account_info()
        self.assertIsNotNone(account_info)
        self.assertEqual(account_info.account_id, "12345")
        self.assertEqual(account_info.balance, 10000.0)
        self.assertEqual(account_info.equity, 10050.0)
        self.assertEqual(account_info.margin_used, 100.0)
        self.assertEqual(account_info.margin_available, 9950.0)
        self.assertEqual(account_info.currency, "USD")
        self.mock_mt5.account_info.assert_called_once()


if __name__ == '__main__':
    unittest.main()
