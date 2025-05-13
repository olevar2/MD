"""
cTrader broker adapter implementation.
"""
import asyncio
import logging
import hmac
import hashlib
import json
import websockets
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
import aiohttp
from .base_broker_adapter import BaseBrokerAdapter
from ..interfaces.broker_adapter import OrderRequest, ExecutionReport, PositionUpdate, AccountUpdate, OrderType, OrderDirection, OrderStatus
logger = logging.getLogger(__name__)
from trading_gateway_service.error.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from trading_gateway_service.resilience.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class CTraderAdapter(BaseBrokerAdapter):
    """
    cTrader broker adapter implementation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cTrader adapter.
        
        Args:
            config: Configuration dictionary containing:
                - api_url: REST API endpoint
                - ws_url: WebSocket endpoint
                - client_id: cTrader client ID
                - client_secret: cTrader client secret
                - account_id: Trading account ID
                Additional base adapter config parameters
        """
        super().__init__(config)
        self.ws_url = config['ws_url']
        self.client_id = config['client_id']
        self.client_secret = config['client_secret']
        self.account_id = config['account_id']
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.ws_task: Optional[asyncio.Task] = None
        self.subscribed_symbols: Set[str] = set()
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        self._refresh_token: Optional[str] = None

    def _get_auth_headers(self) ->Dict[str, str]:
        """Get authentication headers for API requests."""
        if not self._access_token:
            raise Exception('Not authenticated')
        return {'Authorization': f'Bearer {self._access_token}',
            'Content-Type': 'application/json'}

    @async_with_exception_handling
    async def _authenticate(self) ->bool:
        """Authenticate with cTrader API."""
        try:
            async with aiohttp.ClientSession() as session:
                auth_data = {'grant_type': 'client_credentials',
                    'client_id': self.client_id, 'client_secret': self.
                    client_secret, 'scope': 'trading'}
                async with session.post(f'{self.api_url}/connect/token',
                    data=auth_data) as response:
                    if response.status != 200:
                        raise Exception(
                            f'Authentication failed: {response.status}')
                    data = await response.json()
                    self._access_token = data['access_token']
                    self._refresh_token = data.get('refresh_token')
                    expires_in = data.get('expires_in', 3600)
                    self._token_expires = datetime.utcnow().timestamp(
                        ) + expires_in
                    return True
        except Exception as e:
            logger.error(f'Authentication error: {str(e)}')
            return False

    @async_with_exception_handling
    async def connect(self) ->bool:
        """
        Establish connection to cTrader.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if not await self._authenticate():
                return False
            self.ws = await websockets.connect(self.ws_url, extra_headers=
                self._get_auth_headers())
            self.ws_task = asyncio.create_task(self._ws_listener())
            await self._subscribe_account()
            self._is_connected = True
            await self._start_heartbeat()
            logger.info('Successfully connected to cTrader')
            return True
        except Exception as e:
            logger.error(f'Failed to connect to cTrader: {str(e)}')
            return False

    @async_with_exception_handling
    async def disconnect(self) ->bool:
        """
        Disconnect from cTrader.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            self._is_connected = False
            if self.ws_task:
                self.ws_task.cancel()
            if self.ws:
                await self.ws.close()
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            logger.info('Successfully disconnected from cTrader')
            return True
        except Exception as e:
            logger.error(f'Error disconnecting from cTrader: {str(e)}')
            return False

    @async_with_exception_handling
    async def _ws_listener(self) ->None:
        """Listen for WebSocket messages."""
        try:
            while self._is_connected:
                message = await self.ws.recv()
                data = json.loads(message)
                msg_type = data.get('type')
                if msg_type == 'execution':
                    if self._execution_callback:
                        await self._execution_callback(self.
                            _convert_execution(data))
                elif msg_type == 'position':
                    if self._position_callback:
                        await self._position_callback(self.
                            _convert_position(data))
                elif msg_type == 'account':
                    if self._account_callback:
                        await self._account_callback(self._convert_account(
                            data))
                elif msg_type == 'heartbeat':
                    self._last_heartbeat = datetime.utcnow()
        except websockets.exceptions.ConnectionClosed:
            logger.warning('WebSocket connection closed')
            await self._handle_connection_loss()
        except Exception as e:
            logger.error(f'Error in WebSocket listener: {str(e)}')
            await self._handle_connection_loss()

    async def _subscribe_account(self) ->None:
        """Subscribe to account updates."""
        if self.ws:
            await self.ws.send(json.dumps({'type': 'subscribe', 'channel':
                'account', 'accountId': self.account_id}))

    async def _send_heartbeat(self) ->None:
        """Send heartbeat message."""
        if self.ws:
            await self.ws.send(json.dumps({'type': 'ping'}))

    def _convert_execution(self, data: Dict) ->ExecutionReport:
        """Convert cTrader execution message to ExecutionReport."""
        return ExecutionReport(broker_order_id=str(data['orderId']),
            client_order_id=data.get('clientOrderId', ''), instrument=data[
            'symbol'], status=self._convert_status(data['status']),
            filled_quantity=data.get('filledQuantity', 0), average_price=
            data.get('averagePrice'), rejection_reason=data.get('rejectReason')
            )

    def _convert_position(self, data: Dict) ->PositionUpdate:
        """Convert cTrader position message to PositionUpdate."""
        return PositionUpdate(instrument=data['symbol'], position_id=str(
            data['positionId']), quantity=data['volume'], average_price=
            data['openPrice'], unrealized_pl=data['unrealizedPL'],
            margin_used=data.get('margin'))

    def _convert_account(self, data: Dict) ->AccountUpdate:
        """Convert cTrader account message to AccountUpdate."""
        return AccountUpdate(account_id=str(data['accountId']), balance=
            data['balance'], equity=data['equity'], margin_used=data[
            'marginUsed'], margin_available=data['marginAvailable'],
            currency=data['currency'])

    def _convert_status(self, ctrader_status: str) ->OrderStatus:
        """Convert cTrader order status to our OrderStatus."""
        status_map = {'new': OrderStatus.PENDING, 'partiallyFilled':
            OrderStatus.PARTIALLY_FILLED, 'filled': OrderStatus.FILLED,
            'cancelled': OrderStatus.CANCELLED, 'rejected': OrderStatus.
            REJECTED, 'expired': OrderStatus.EXPIRED}
        return status_map.get(ctrader_status, OrderStatus.REJECTED)

    @async_with_exception_handling
    async def place_order(self, order_request: OrderRequest) ->ExecutionReport:
        """
        Place a new order with cTrader.
        
        Args:
            order_request: The order to place
            
        Returns:
            Execution report for the order
        """
        if not self._is_connected:
            self._order_queue.append(order_request)
            return ExecutionReport(broker_order_id='', client_order_id=
                order_request.client_order_id, instrument=order_request.
                instrument, status=OrderStatus.PENDING, rejection_reason=
                'Not connected - order queued')
        try:
            request_data = {'symbol': order_request.instrument, 'type':
                order_request.order_type.value, 'side': order_request.
                direction.value, 'quantity': order_request.quantity,
                'clientOrderId': order_request.client_order_id}
            if order_request.order_type != OrderType.MARKET:
                request_data['price'] = order_request.price
            if order_request.stop_loss:
                request_data['stopLoss'] = order_request.stop_loss
            if order_request.take_profit:
                request_data['takeProfit'] = order_request.take_profit
            async with aiohttp.ClientSession(headers=self._get_auth_headers()
                ) as session:
                async with session.post(f'{self.api_url}/v1/orders', json=
                    request_data) as response:
                    data = await response.json()
                    if response.status != 201:
                        return ExecutionReport(broker_order_id='',
                            client_order_id=order_request.client_order_id,
                            instrument=order_request.instrument, status=
                            OrderStatus.REJECTED, rejection_reason=data.get
                            ('message', 'Unknown error'))
                    return self._convert_execution(data)
        except Exception as e:
            logger.error(f'Error placing order: {str(e)}')
            return ExecutionReport(broker_order_id='', client_order_id=
                order_request.client_order_id, instrument=order_request.
                instrument, status=OrderStatus.REJECTED, rejection_reason=
                str(e))

    @async_with_exception_handling
    async def cancel_order(self, client_order_id: str) ->ExecutionReport:
        """
        Cancel an existing order.
        
        Args:
            client_order_id: The client order ID to cancel
            
        Returns:
            Execution report indicating the cancellation status
        """
        try:
            async with aiohttp.ClientSession(headers=self._get_auth_headers()
                ) as session:
                async with session.delete(f'{self.api_url}/v1/orders',
                    params={'clientOrderId': client_order_id}) as response:
                    if response.status != 200:
                        data = await response.json()
                        return ExecutionReport(broker_order_id='',
                            client_order_id=client_order_id, instrument='',
                            status=OrderStatus.REJECTED, rejection_reason=
                            data.get('message', 'Cancel failed'))
                    return ExecutionReport(broker_order_id='',
                        client_order_id=client_order_id, instrument='',
                        status=OrderStatus.CANCELLED)
        except Exception as e:
            logger.error(f'Error cancelling order: {str(e)}')
            return ExecutionReport(broker_order_id='', client_order_id=
                client_order_id, instrument='', status=OrderStatus.REJECTED,
                rejection_reason=str(e))

    @async_with_exception_handling
    async def modify_order(self, client_order_id: str, modifications: Dict[
        str, Any]) ->ExecutionReport:
        """
        Modify an existing order.
        
        Args:
            client_order_id: The client order ID to modify
            modifications: Dictionary of fields to modify
            
        Returns:
            Execution report indicating the modification status
        """
        try:
            async with aiohttp.ClientSession(headers=self._get_auth_headers()
                ) as session:
                async with session.patch(
                    f'{self.api_url}/v1/orders/{client_order_id}', json=
                    modifications) as response:
                    data = await response.json()
                    if response.status != 200:
                        return ExecutionReport(broker_order_id='',
                            client_order_id=client_order_id, instrument='',
                            status=OrderStatus.REJECTED, rejection_reason=
                            data.get('message', 'Modification failed'))
                    return self._convert_execution(data)
        except Exception as e:
            logger.error(f'Error modifying order: {str(e)}')
            return ExecutionReport(broker_order_id='', client_order_id=
                client_order_id, instrument='', status=OrderStatus.REJECTED,
                rejection_reason=str(e))

    @with_broker_api_resilience('get_positions')
    @async_with_exception_handling
    async def get_positions(self) ->List[PositionUpdate]:
        """
        Get all current positions.
        
        Returns:
            List of position updates
        """
        try:
            async with aiohttp.ClientSession(headers=self._get_auth_headers()
                ) as session:
                async with session.get(f'{self.api_url}/v1/positions'
                    ) as response:
                    if response.status != 200:
                        return []
                    data = await response.json()
                    return [self._convert_position(pos) for pos in data]
        except Exception as e:
            logger.error(f'Error getting positions: {str(e)}')
            return []

    @with_broker_api_resilience('get_account_info')
    @async_with_exception_handling
    async def get_account_info(self) ->AccountUpdate:
        """
        Get current account information.
        
        Returns:
            Account update with current information
        """
        try:
            async with aiohttp.ClientSession(headers=self._get_auth_headers()
                ) as session:
                async with session.get(
                    f'{self.api_url}/v1/accounts/{self.account_id}'
                    ) as response:
                    if response.status != 200:
                        raise Exception('Failed to get account information')
                    data = await response.json()
                    return self._convert_account(data)
        except Exception as e:
            logger.error(f'Error getting account info: {str(e)}')
            raise

    @async_with_exception_handling
    async def close_position(self, position_id: str, quantity: Optional[
        float]=None) ->ExecutionReport:
        """
        Close an existing position.
        
        Args:
            position_id: The position ID to close
            quantity: Optional quantity to close (if None, close entire position)
            
        Returns:
            Execution report for the closing order
        """
        try:
            request_data = {'positionId': position_id, 'quantity': quantity}
            async with aiohttp.ClientSession(headers=self._get_auth_headers()
                ) as session:
                async with session.post(f'{self.api_url}/v1/positions/close',
                    json=request_data) as response:
                    data = await response.json()
                    if response.status != 200:
                        return ExecutionReport(broker_order_id='',
                            client_order_id='', instrument='', status=
                            OrderStatus.REJECTED, rejection_reason=data.get
                            ('message', 'Close failed'))
                    return self._convert_execution(data)
        except Exception as e:
            logger.error(f'Error closing position: {str(e)}')
            return ExecutionReport(broker_order_id='', client_order_id='',
                instrument='', status=OrderStatus.REJECTED,
                rejection_reason=str(e))

    @async_with_exception_handling
    async def subscribe_to_updates(self, callback_execution: callable,
        callback_position: callable, callback_account: callable) ->bool:
        """
        Subscribe to real-time updates.
        
        Args:
            callback_execution: Callback for execution updates
            callback_position: Callback for position updates
            callback_account: Callback for account updates
            
        Returns:
            True if subscription successful, False otherwise
        """
        try:
            self._execution_callback = callback_execution
            self._position_callback = callback_position
            self._account_callback = callback_account
            return True
        except Exception as e:
            logger.error(f'Error subscribing to updates: {str(e)}')
            return False

    async def unsubscribe_from_updates(self) ->bool:
        """
        Unsubscribe from real-time updates.
        
        Returns:
            True if unsubscription successful, False otherwise
        """
        self._execution_callback = None
        self._position_callback = None
        self._account_callback = None
        return True

    @property
    def name(self) ->str:
        """Get the name of the broker."""
        return 'cTrader'
