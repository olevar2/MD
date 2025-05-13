"""
Unit tests for the Interactive Brokers adapter.

This module contains tests for the InteractiveBrokersAdapter class, which provides
connectivity to Interactive Brokers using the TWS API.
"""

import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from datetime import datetime
import time
import threading
from queue import Queue

from adapters.interactive_brokers_adapter import (
    InteractiveBrokersAdapter, IBWrapperImpl
)
from adapters.broker_adapter import (
    OrderRequest, OrderStatus, OrderDirection, OrderType
)


class TestInteractiveBrokersAdapter(unittest.TestCase):
    """Tests for the Interactive Brokers adapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Patch the IB API classes
        self.eclient_patcher = patch('trading_gateway_service.broker_adapters.interactive_brokers_adapter.EClient')
        self.ewrapper_patcher = patch('trading_gateway_service.broker_adapters.interactive_brokers_adapter.EWrapper')
        self.contract_patcher = patch('trading_gateway_service.broker_adapters.interactive_brokers_adapter.Contract')
        self.iborder_patcher = patch('trading_gateway_service.broker_adapters.interactive_brokers_adapter.IBOrder')
        
        self.mock_eclient = self.eclient_patcher.start()
        self.mock_ewrapper = self.ewrapper_patcher.start()
        self.mock_contract = self.contract_patcher.start()
        self.mock_iborder = self.iborder_patcher.start()
        
        # Create mock instances
        self.mock_eclient_instance = MagicMock()
        self.mock_eclient.return_value = self.mock_eclient_instance
        self.mock_eclient_instance.isConnected.return_value = True
        
        self.mock_contract_instance = MagicMock()
        self.mock_contract.return_value = self.mock_contract_instance
        
        self.mock_iborder_instance = MagicMock()
        self.mock_iborder.return_value = self.mock_iborder_instance
        
        # Create adapter with test configuration
        self.config = {
            'host': '127.0.0.1',
            'port': 7497,
            'client_id': 1,
            'connect_timeout': 5
        }
        
        # Create adapter
        self.adapter = InteractiveBrokersAdapter(self.config)
        self.adapter._client = self.mock_eclient_instance
        
        # Mock the wrapper's response queue
        self.adapter._wrapper.response_queue = Queue()
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.eclient_patcher.stop()
        self.ewrapper_patcher.stop()
        self.contract_patcher.stop()
        self.iborder_patcher.stop()
    
    async def test_connect(self):
        """Test connecting to Interactive Brokers."""
        # Setup
        self.adapter._connected = False
        self.adapter._next_order_id = None
        
        # Mock the thread creation
        with patch('threading.Thread') as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance
            
            # Put a successful connection response in the queue
            self.adapter._wrapper.response_queue.put(("connected", True))
            
            # Test
            result = await self.adapter.connect()
            self.assertTrue(result)
            self.assertTrue(self.adapter._connected)
            self.mock_eclient_instance.connect.assert_called_once_with(
                self.config['host'], self.config['port'], self.config['client_id']
            )
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()
    
    async def test_disconnect(self):
        """Test disconnecting from Interactive Brokers."""
        # Setup
        self.adapter._connected = True
        self.adapter._api_thread = MagicMock()
        
        # Test
        result = await self.adapter.disconnect()
        self.assertTrue(result)
        self.assertFalse(self.adapter._connected)
        self.mock_eclient_instance.disconnect.assert_called_once()
    
    async def test_place_order(self):
        """Test placing an order."""
        # Setup
        self.adapter._connected = True
        self.adapter._next_order_id = 12345
        
        order_request = OrderRequest(
            instrument="EUR_USD",
            order_type=OrderType.MARKET,
            direction=OrderDirection.BUY,
            quantity=10000,
            price=1.1000,
            client_order_id="test-order-1"
        )
        
        # Test
        result = await self.adapter.place_order(order_request)
        self.assertEqual(result.status, OrderStatus.PENDING)
        self.assertEqual(result.broker_order_id, "12345")
        self.assertEqual(result.client_order_id, "test-order-1")
        self.assertEqual(result.instrument, "EUR_USD")
        self.assertEqual(result.quantity, 10000)
        
        # Verify contract creation
        self.mock_contract.assert_called_once()
        self.mock_contract_instance.symbol = "EUR"
        self.mock_contract_instance.secType = "CASH"
        self.mock_contract_instance.currency = "USD"
        self.mock_contract_instance.exchange = "IDEALPRO"
        
        # Verify order creation
        self.mock_iborder.assert_called_once()
        self.mock_iborder_instance.orderId = 12345
        self.mock_iborder_instance.action = "BUY"
        self.mock_iborder_instance.orderType = "MKT"
        self.mock_iborder_instance.totalQuantity = 10000
        self.mock_iborder_instance.transmit = True
        
        # Verify order placement
        self.mock_eclient_instance.placeOrder.assert_called_once_with(
            12345, self.mock_contract_instance, self.mock_iborder_instance
        )
    
    async def test_cancel_order(self):
        """Test cancelling an order."""
        # Setup
        self.adapter._connected = True
        client_order_id = "test-order-1"
        broker_order_id = "12345"
        
        # Add mapping
        self.adapter._order_map[client_order_id] = broker_order_id
        self.adapter._reverse_order_map[int(broker_order_id)] = client_order_id
        
        # Test
        result = await self.adapter.cancel_order(client_order_id, broker_order_id)
        self.assertEqual(result.status, OrderStatus.PENDING)
        self.assertEqual(result.broker_order_id, "12345")
        self.assertEqual(result.client_order_id, "test-order-1")
        
        # Verify cancel order call
        self.mock_eclient_instance.cancelOrder.assert_called_once_with(int(broker_order_id))
    
    def test_on_order_status(self):
        """Test handling order status updates."""
        # Setup
        order_id = 12345
        client_order_id = "test-order-1"
        self.adapter._reverse_order_map[order_id] = client_order_id
        
        # Register a callback
        callback_called = False
        callback_report = None
        
        async def mock_callback(report):
    """
    Mock callback.
    
    Args:
        report: Description of report
    
    """

            nonlocal callback_called, callback_report
            callback_called = True
            callback_report = report
            
        self.adapter._callback_execution = mock_callback
        
        # Test
        self.adapter._on_order_status(
            order_id, "Filled", 10000, 0, 1.1000, 67890
        )
        
        # Verify callback was called with correct report
        self.assertTrue(callback_called)
        self.assertEqual(callback_report.status, OrderStatus.FILLED)
        self.assertEqual(callback_report.broker_order_id, "12345")
        self.assertEqual(callback_report.client_order_id, "test-order-1")
        self.assertEqual(callback_report.filled_quantity, 10000)
        self.assertEqual(callback_report.average_price, 1.1000)
    
    def test_wrapper_next_valid_id(self):
        """Test handling nextValidId callback."""
        # Setup
        wrapper = IBWrapperImpl(self.adapter)
        
        # Test
        wrapper.nextValidId(12345)
        
        # Verify
        self.assertEqual(self.adapter._next_order_id, 12345)
        self.assertEqual(wrapper.response_queue.get(), ("connected", True))


if __name__ == '__main__':
    unittest.main()
