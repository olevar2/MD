"""
Unit tests for the Oanda adapter.

This module contains tests for the OandaAdapter class, which provides
connectivity to Oanda's REST API v20.
"""

import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from datetime import datetime
import httpx

from adapters.oanda_adapter import OandaAdapter
from adapters.broker_adapter import (
    OrderRequest, OrderStatus, OrderDirection, OrderType
)


class TestOandaAdapter(unittest.TestCase):
    """Tests for the Oanda adapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create adapter with test configuration
        self.config = {
            'account_id': '001-001-12345678-001',
            'access_token': 'test_access_token',
            'environment': 'practice',
            'request_timeout': 5
        }
        
        # Patch httpx.AsyncClient
        self.client_patcher = patch('httpx.AsyncClient')
        self.mock_client_class = self.client_patcher.start()
        self.mock_client = AsyncMock()
        self.mock_client_class.return_value = self.mock_client
        
        # Create adapter
        self.adapter = OandaAdapter(self.config)
        self.adapter._client = self.mock_client
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.client_patcher.stop()
    
    async def test_connect(self):
        """Test connecting to Oanda."""
        # Setup
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'account': {
                'id': '001-001-12345678-001',
                'currency': 'USD',
                'balance': '10000.00',
                'marginRate': '0.02'
            }
        }
        mock_response.raise_for_status = MagicMock()
        self.mock_client.get.return_value = mock_response
        
        # Test
        result = await self.adapter.connect()
        self.assertTrue(result)
        self.assertTrue(self.adapter._connected)
        self.mock_client.get.assert_called_once_with(
            f"/v3/accounts/{self.config['account_id']}/summary"
        )
    
    async def test_disconnect(self):
        """Test disconnecting from Oanda."""
        # Setup
        self.adapter._connected = True
        self.adapter._streaming_task = None
        
        # Test
        result = await self.adapter.disconnect()
        self.assertTrue(result)
        self.assertFalse(self.adapter._connected)
        self.mock_client.aclose.assert_called_once()
    
    async def test_place_order(self):
        """Test placing an order."""
        # Setup
        self.adapter._connected = True
        order_request = OrderRequest(
            instrument="EUR_USD",
            order_type=OrderType.MARKET,
            direction=OrderDirection.BUY,
            quantity=10000,
            price=1.1000,
            client_order_id="test-order-1",
            extra_data={"strategy_id": "test-strategy"}
        )
        
        # Mock response for successful order placement
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            'orderCreateTransaction': {
                'id': '12345',
                'accountID': '001-001-12345678-001',
                'batchID': '12345',
                'requestID': 'test-request',
                'time': '2023-01-01T00:00:00.000000000Z',
                'type': 'MARKET_ORDER',
                'instrument': 'EUR_USD',
                'units': '10000',
                'timeInForce': 'FOK',
                'positionFill': 'DEFAULT',
                'clientExtensions': {
                    'id': 'test-order-1',
                    'comment': 'Strategy: test-strategy'
                }
            },
            'orderFillTransaction': {
                'id': '12346',
                'accountID': '001-001-12345678-001',
                'batchID': '12345',
                'requestID': 'test-request',
                'time': '2023-01-01T00:00:00.000000000Z',
                'type': 'ORDER_FILL',
                'orderID': '12345',
                'instrument': 'EUR_USD',
                'units': '10000',
                'price': '1.10000',
                'pl': '0.0000',
                'financing': '0.0000',
                'commission': '0.0000',
                'accountBalance': '10000.0000',
                'tradeOpened': {
                    'tradeID': '12347',
                    'units': '10000'
                }
            },
            'relatedTransactionIDs': ['12345', '12346'],
            'lastTransactionID': '12346'
        }
        self.mock_client.post.return_value = mock_response
        
        # Test
        result = await self.adapter.place_order(order_request)
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertEqual(result.broker_order_id, "12345")
        self.assertEqual(result.client_order_id, "test-order-1")
        self.assertEqual(result.instrument, "EUR_USD")
        self.assertEqual(result.filled_quantity, 10000)
        self.assertEqual(result.average_price, 1.1000)
        self.mock_client.post.assert_called_once()
    
    async def test_cancel_order(self):
        """Test cancelling an order."""
        # Setup
        self.adapter._connected = True
        client_order_id = "test-order-1"
        broker_order_id = "12345"
        
        # Mock response for successful order cancellation
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'orderCancelTransaction': {
                'id': '12347',
                'accountID': '001-001-12345678-001',
                'batchID': '12347',
                'requestID': 'test-request',
                'time': '2023-01-01T00:00:00.000000000Z',
                'type': 'ORDER_CANCEL',
                'orderID': '12345',
                'reason': 'CLIENT_REQUEST'
            },
            'relatedTransactionIDs': ['12347'],
            'lastTransactionID': '12347'
        }
        self.mock_client.put.return_value = mock_response
        
        # Test
        result = await self.adapter.cancel_order(client_order_id, broker_order_id)
        self.assertEqual(result.status, OrderStatus.CANCELLED)
        self.assertEqual(result.broker_order_id, "12345")
        self.mock_client.put.assert_called_once_with(
            f"/v3/accounts/{self.config['account_id']}/orders/@{broker_order_id}/cancel"
        )
    
    async def test_get_positions(self):
        """Test getting positions."""
        # Setup
        self.adapter._connected = True
        
        # Mock response for positions
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'positions': [
                {
                    'instrument': 'EUR_USD',
                    'long': {
                        'units': '10000',
                        'averagePrice': '1.10000',
                        'pl': '50.0000',
                        'unrealizedPL': '50.0000',
                        'resettablePL': '50.0000'
                    },
                    'short': {
                        'units': '0',
                        'averagePrice': '0.00000',
                        'pl': '0.0000',
                        'unrealizedPL': '0.0000',
                        'resettablePL': '0.0000'
                    },
                    'pl': '50.0000',
                    'unrealizedPL': '50.0000',
                    'resettablePL': '50.0000'
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        self.mock_client.get.return_value = mock_response
        
        # Test
        positions = await self.adapter.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].instrument, "EUR_USD")
        self.assertEqual(positions[0].quantity, 10000)
        self.assertEqual(positions[0].average_price, 1.1000)
        self.assertEqual(positions[0].unrealized_pl, 50.0)
        self.mock_client.get.assert_called_once_with(
            f"/v3/accounts/{self.config['account_id']}/openPositions"
        )
    
    async def test_get_account_info(self):
        """Test getting account information."""
        # Setup
        self.adapter._connected = True
        
        # Mock response for account info
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'account': {
                'id': '001-001-12345678-001',
                'currency': 'USD',
                'balance': '10000.0000',
                'openTradeCount': 1,
                'openPositionCount': 1,
                'pendingOrderCount': 0,
                'pl': '50.0000',
                'unrealizedPL': '50.0000',
                'resettablePL': '50.0000',
                'financing': '0.0000',
                'commission': '0.0000',
                'marginRate': '0.02',
                'marginCallMarginUsed': '100.0000',
                'marginCallPercent': '0.05',
                'marginCloseoutMarginUsed': '100.0000',
                'marginCloseoutPercent': '0.05',
                'withdrawalLimit': '9950.0000',
                'marginAvailable': '9950.0000',
                'positionValue': '11000.0000',
                'marginUsed': '100.0000'
            }
        }
        mock_response.raise_for_status = MagicMock()
        self.mock_client.get.return_value = mock_response
        
        # Test
        account_info = await self.adapter.get_account_info()
        self.assertIsNotNone(account_info)
        self.assertEqual(account_info.account_id, "001-001-12345678-001")
        self.assertEqual(account_info.balance, 10000.0)
        self.assertEqual(account_info.equity, 10050.0)  # balance + unrealizedPL
        self.assertEqual(account_info.margin_used, 100.0)
        self.assertEqual(account_info.margin_available, 9950.0)
        self.assertEqual(account_info.currency, "USD")
        self.mock_client.get.assert_called_once_with(
            f"/v3/accounts/{self.config['account_id']}/summary"
        )


if __name__ == '__main__':
    unittest.main()
