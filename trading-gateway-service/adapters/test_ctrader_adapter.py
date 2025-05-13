"""
Unit tests for the cTrader adapter.

This module contains tests for the CTraderAdapter class, which provides
connectivity to cTrader platforms using the Open API.
"""

import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from datetime import datetime

from adapters.ctrader_adapter import CTraderAdapter
from adapters.broker_adapter import (
    OrderRequest, OrderStatus, OrderDirection, OrderType
)


class TestCTraderAdapter(unittest.TestCase):
    """Tests for the cTrader adapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create adapter with test configuration
        self.config = {
            'host': 'demo.ctrader.com',
            'port': 5035,
            'client_id': 'test_client_id',
            'client_secret': 'test_client_secret',
            'account_id': '12345',
            'request_timeout': 5
        }
        
        # Patch asyncio.open_connection
        self.open_connection_patcher = patch('asyncio.open_connection')
        self.mock_open_connection = self.open_connection_patcher.start()
        
        # Mock StreamReader and StreamWriter
        self.mock_reader = AsyncMock()
        self.mock_writer = AsyncMock()
        self.mock_open_connection.return_value = (self.mock_reader, self.mock_writer)
        
        # Create adapter
        self.adapter = CTraderAdapter(self.config)
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.open_connection_patcher.stop()
    
    async def test_connect(self):
        """Test connecting to cTrader."""
        # Setup
        self.mock_reader.readexactly.return_value = b'\x00\x00\x00\x10'  # Mock message length
        self.mock_reader.readexactly.return_value = b'\x00\x00\x00\x00'  # Mock message payload
        
        # Test
        result = await self.adapter.connect()
        self.assertTrue(result)
        self.assertTrue(self.adapter.is_connected)
        self.mock_open_connection.assert_called_once_with(
            self.config['host'], self.config['port']
        )
    
    async def test_disconnect(self):
        """Test disconnecting from cTrader."""
        # Setup
        self.adapter._connected = True
        self.adapter._writer = self.mock_writer
        self.adapter._listener_task = asyncio.create_task(asyncio.sleep(0))
        
        # Test
        result = await self.adapter.disconnect()
        self.assertTrue(result)
        self.assertFalse(self.adapter.is_connected)
        self.mock_writer.close.assert_called_once()
        self.mock_writer.wait_closed.assert_called_once()
    
    async def test_place_order(self):
        """Test placing an order."""
        # Setup
        self.adapter._connected = True
        self.adapter._account_id = 12345
        self.adapter._writer = self.mock_writer
        
        order_request = OrderRequest(
            instrument="EURUSD",
            order_type=OrderType.MARKET,
            direction=OrderDirection.BUY,
            quantity=0.1,
            price=1.1000,
            client_order_id="test-order-1"
        )
        
        # Mock _get_symbol_id and _convert_quantity_to_volume
        self.adapter._get_symbol_id = AsyncMock(return_value=1)
        self.adapter._convert_quantity_to_volume = AsyncMock(return_value=10000)
        
        # Mock _send_message
        self.adapter._send_message = AsyncMock()
        
        # Create a future that will be set by the test
        future = asyncio.Future()
        future.set_result({"status": "ACCEPTED"})
        
        # Mock the pending_requests dictionary
        self.adapter._pending_requests = {}
        self.adapter._get_next_client_msg_id = MagicMock(return_value=1)
        
        # Patch asyncio.get_event_loop().create_future
        with patch('asyncio.get_event_loop') as mock_get_event_loop:
            mock_loop = MagicMock()
            mock_loop.create_future.return_value = future
            mock_get_event_loop.return_value = mock_loop
            
            # Test
            result = await self.adapter.place_order(order_request)
            self.assertEqual(result.status, OrderStatus.ACCEPTED)
            self.assertEqual(result.client_order_id, "test-order-1")
            self.assertEqual(result.instrument, "EURUSD")
    
    async def test_cancel_order(self):
        """Test cancelling an order."""
        # Setup
        self.adapter._connected = True
        self.adapter._account_id = 12345
        self.adapter._writer = self.mock_writer
        
        client_order_id = "test-order-1"
        broker_order_id = "12345"
        
        # Mock _send_message
        self.adapter._send_message = AsyncMock()
        
        # Create a future that will be set by the test
        future = asyncio.Future()
        future.set_result({"status": "CANCELLED"})
        
        # Mock the pending_requests dictionary
        self.adapter._pending_requests = {}
        self.adapter._get_next_client_msg_id = MagicMock(return_value=1)
        
        # Patch asyncio.get_event_loop().create_future
        with patch('asyncio.get_event_loop') as mock_get_event_loop:
            mock_loop = MagicMock()
            mock_loop.create_future.return_value = future
            mock_get_event_loop.return_value = mock_loop
            
            # Test
            result = await self.adapter.cancel_order(client_order_id, broker_order_id)
            self.assertEqual(result.status, OrderStatus.CANCELLED)
            self.assertEqual(result.client_order_id, "test-order-1")
            self.assertEqual(result.broker_order_id, "12345")
    
    async def test_get_positions(self):
        """Test getting positions."""
        # Setup
        self.adapter._connected = True
        self.adapter._account_id = 12345
        self.adapter._writer = self.mock_writer
        
        # Mock _send_message
        self.adapter._send_message = AsyncMock()
        
        # Create a future that will be set by the test
        future = asyncio.Future()
        future.set_result({"positions": [
            {
                "positionId": 12345,
                "symbol": "EURUSD",
                "volume": 0.1,
                "entryPrice": 1.1000,
                "unrealizedGrossProfit": 5.0,
                "usedMargin": 10.0
            }
        ]})
        
        # Mock the pending_requests dictionary
        self.adapter._pending_requests = {}
        self.adapter._get_next_client_msg_id = MagicMock(return_value=1)
        
        # Patch asyncio.get_event_loop().create_future
        with patch('asyncio.get_event_loop') as mock_get_event_loop:
            mock_loop = MagicMock()
            mock_loop.create_future.return_value = future
            mock_get_event_loop.return_value = mock_loop
            
            # Test
            positions = await self.adapter.get_positions()
            self.assertEqual(len(positions), 1)
            self.assertEqual(positions[0].instrument, "EURUSD_placeholder")
            self.assertEqual(positions[0].position_id, "CT_POS_ID_1")
            self.assertEqual(positions[0].quantity, 0.1)
            self.assertEqual(positions[0].average_price, 1.1000)
            self.assertEqual(positions[0].unrealized_pl, 10.5)
    
    async def test_get_account_info(self):
        """Test getting account information."""
        # Setup
        self.adapter._connected = True
        self.adapter._account_id = 12345
        self.adapter._writer = self.mock_writer
        
        # Mock _send_message
        self.adapter._send_message = AsyncMock()
        
        # Create a future that will be set by the test
        future = asyncio.Future()
        future.set_result({"account": {
            "accountId": 12345,
            "balance": 10000.0,
            "equity": 10050.0,
            "marginUsed": 100.0,
            "freeMargin": 9950.0,
            "currency": "USD"
        }})
        
        # Mock the pending_requests dictionary
        self.adapter._pending_requests = {}
        self.adapter._get_next_client_msg_id = MagicMock(return_value=1)
        
        # Patch asyncio.get_event_loop().create_future
        with patch('asyncio.get_event_loop') as mock_get_event_loop:
            mock_loop = MagicMock()
            mock_loop.create_future.return_value = future
            mock_get_event_loop.return_value = mock_loop
            
            # Test
            account_info = await self.adapter.get_account_info()
            self.assertIsNotNone(account_info)
            self.assertEqual(account_info.account_id, "12345")
            self.assertEqual(account_info.balance, 10000.0)
            self.assertEqual(account_info.equity, 10100.50)
            self.assertEqual(account_info.margin_used, 50.0)
            self.assertEqual(account_info.margin_available, 9950.0)
            self.assertEqual(account_info.currency, "USD")


if __name__ == '__main__':
    unittest.main()
