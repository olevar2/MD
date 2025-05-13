"""
OANDA broker adapter implementation.
"""
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
import aiohttp
from yarl import URL
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

class OandaAdapter(BaseBrokerAdapter):
    """
    OANDA broker adapter implementation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OANDA adapter.
        
        Args:
            config: Configuration dictionary containing:
                - api_url: API endpoint (practice/live)
                - stream_url: Streaming API endpoint
                - access_token: OANDA API access token
                - account_id: Trading account ID
                - datetime_format: Optional datetime format (default: RFC3339)
                Additional base adapter config parameters
        """
        super().__init__(config)
        self.stream_url = config['stream_url']
        self.access_token = config['access_token']
        self.account_id = config['account_id']
        self.datetime_format = config_manager.get('datetime_format', 'UNIX')
        self.stream_client: Optional[aiohttp.ClientSession] = None
        self.stream_response: Optional[aiohttp.ClientResponse] = None
        self.stream_task: Optional[asyncio.Task] = None
        self.price_streams: Set[str] = set()

    def _get_auth_headers(self) ->Dict[str, str]:
        """Get authentication headers for API requests."""
        return {'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json', 'Accept-Datetime-Format':
            self.datetime_format}

    @async_with_exception_handling
    async def connect(self) ->bool:
        """
        Establish connection to OANDA.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            async with aiohttp.ClientSession(headers=self._get_auth_headers()
                ) as session:
                async with session.get(URL(self.api_url).join(
                    f'/v3/accounts/{self.account_id}')) as response:
                    if response.status != 200:
                        raise Exception(f'Connection failed: {response.status}'
                            )
                    await response.json()
            self.stream_client = aiohttp.ClientSession(headers=self.
                _get_auth_headers())
            await self._start_streams()
            self._is_connected = True
            await self._start_heartbeat()
            logger.info('Successfully connected to OANDA')
            return True
        except Exception as e:
            logger.error(f'Failed to connect to OANDA: {str(e)}')
            if self.stream_client:
                await self.stream_client.close()
            return False

    @async_with_exception_handling
    async def disconnect(self) ->bool:
        """
        Disconnect from OANDA.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            self._is_connected = False
            if self.stream_task:
                self.stream_task.cancel()
            if self.stream_response:
                await self.stream_response.release()
            if self.stream_client:
                await self.stream_client.close()
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            logger.info('Successfully disconnected from OANDA')
            return True
        except Exception as e:
            logger.error(f'Error disconnecting from OANDA: {str(e)}')
            return False

    async def _start_streams(self) ->None:
        """Initialize price and transaction streams."""
        transactions_url = URL(self.stream_url).join(
            f'/v3/accounts/{self.account_id}/transactions/stream')
        self.stream_response = await self.stream_client.get(transactions_url)
        self.stream_task = asyncio.create_task(self._stream_listener())

    @async_with_exception_handling
    async def _stream_listener(self) ->None:
        """Listen for streaming updates."""
        try:
            async for line in self.stream_response.content:
                if line:
                    try:
                        data = json.loads(line)
                        if 'type' in data:
                            if data['type'] == 'HEARTBEAT':
                                self._last_heartbeat = datetime.utcnow()
                            elif data['type'].startswith('ORDER_'):
                                if self._execution_callback:
                                    await self._execution_callback(self.
                                        _convert_order_update(data))
                            elif data['type'] == 'POSITION':
                                if self._position_callback:
                                    await self._position_callback(self.
                                        _convert_position_update(data))
                            elif data['type'] == 'ACCOUNT':
                                if self._account_callback:
                                    await self._account_callback(self.
                                        _convert_account_update(data))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f'Error in stream listener: {str(e)}')
            await self._handle_connection_loss()

    async def _send_heartbeat(self) ->None:
        """Check stream connection health."""
        if not self.stream_response or self.stream_response.closed:
            raise Exception('Stream connection lost')

    def _convert_order_update(self, data: Dict) ->ExecutionReport:
        """Convert OANDA order update to ExecutionReport."""
        status_map = {'PENDING': OrderStatus.PENDING, 'FILLED': OrderStatus
            .FILLED, 'TRIGGERED': OrderStatus.FILLED, 'CANCELLED':
            OrderStatus.CANCELLED, 'REJECTED': OrderStatus.REJECTED}
        return ExecutionReport(broker_order_id=data['id'], client_order_id=
            data.get('clientOrderID', ''), instrument=data['instrument'],
            status=status_map.get(data['state'], OrderStatus.REJECTED),
            filled_quantity=float(data.get('units', 0)), average_price=
            float(data['price']) if 'price' in data else None,
            rejection_reason=data.get('rejectReason'))

    def _convert_position_update(self, data: Dict) ->PositionUpdate:
        """Convert OANDA position update to PositionUpdate."""
        long_units = float(data.get('long', {}).get('units', 0))
        short_units = float(data.get('short', {}).get('units', 0))
        net_units = long_units + short_units
        long_price = float(data.get('long', {}).get('averagePrice', 0))
        short_price = float(data.get('short', {}).get('averagePrice', 0))
        avg_price = (abs(long_units) * long_price + abs(short_units) *
            short_price) / (abs(long_units) + abs(short_units)
            ) if net_units != 0 else 0
        return PositionUpdate(instrument=data['instrument'], position_id=
            f"{data['instrument']}_{self.account_id}", quantity=net_units,
            average_price=avg_price, unrealized_pl=float(data.get(
            'unrealizedPL', 0)), margin_used=float(data.get('marginUsed', 0)))

    def _convert_account_update(self, data: Dict) ->AccountUpdate:
        """Convert OANDA account update to AccountUpdate."""
        return AccountUpdate(account_id=data['id'], balance=float(data[
            'balance']), equity=float(data.get('NAV', data['balance'])),
            margin_used=float(data['marginUsed']), margin_available=float(
            data['marginAvailable']), currency=data['currency'])

    @async_with_exception_handling
    async def place_order(self, order_request: OrderRequest) ->ExecutionReport:
        """
        Place a new order with OANDA.
        
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
            request_data = {'order': {'type': self._convert_order_type(
                order_request.order_type), 'instrument': order_request.
                instrument, 'units': order_request.quantity if 
                order_request.direction == OrderDirection.BUY else -
                order_request.quantity, 'clientExtensions': {'id':
                order_request.client_order_id}}}
            if order_request.order_type != OrderType.MARKET:
                request_data['order']['price'] = str(order_request.price)
            if order_request.stop_loss:
                request_data['order']['stopLossOnFill'] = {'price': str(
                    order_request.stop_loss)}
            if order_request.take_profit:
                request_data['order']['takeProfitOnFill'] = {'price': str(
                    order_request.take_profit)}
            async with aiohttp.ClientSession(headers=self._get_auth_headers()
                ) as session:
                async with session.post(URL(self.api_url).join(
                    f'/v3/accounts/{self.account_id}/orders'), json=
                    request_data) as response:
                    data = await response.json()
                    if response.status != 201:
                        return ExecutionReport(broker_order_id='',
                            client_order_id=order_request.client_order_id,
                            instrument=order_request.instrument, status=
                            OrderStatus.REJECTED, rejection_reason=str(data
                            .get('errorMessage', 'Unknown error')))
                    order_data = data['orderFillTransaction'
                        ] if 'orderFillTransaction' in data else data[
                        'orderCreateTransaction']
                    return self._convert_order_update(order_data)
        except Exception as e:
            logger.error(f'Error placing order: {str(e)}')
            return ExecutionReport(broker_order_id='', client_order_id=
                order_request.client_order_id, instrument=order_request.
                instrument, status=OrderStatus.REJECTED, rejection_reason=
                str(e))

    def _convert_order_type(self, order_type: OrderType) ->str:
        """Convert internal order type to OANDA order type."""
        type_map = {OrderType.MARKET: 'MARKET', OrderType.LIMIT: 'LIMIT',
            OrderType.STOP: 'STOP', OrderType.STOP_LIMIT: 'STOP_LIMIT'}
        return type_map.get(order_type, 'MARKET')

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
                async with session.get(URL(self.api_url).join(
                    f'/v3/accounts/{self.account_id}/orders')) as response:
                    data = await response.json()
                    order = next((o for o in data['orders'] if o.get(
                        'clientExtensions', {}).get('id') ==
                        client_order_id), None)
                    if not order:
                        return ExecutionReport(broker_order_id='',
                            client_order_id=client_order_id, instrument='',
                            status=OrderStatus.REJECTED, rejection_reason=
                            'Order not found')
                    async with session.put(URL(self.api_url).join(
                        f"/v3/accounts/{self.account_id}/orders/{order['id']}/cancel"
                        )) as cancel_response:
                        cancel_data = await cancel_response.json()
                        if cancel_response.status != 200:
                            return ExecutionReport(broker_order_id=order[
                                'id'], client_order_id=client_order_id,
                                instrument=order['instrument'], status=
                                OrderStatus.REJECTED, rejection_reason=str(
                                cancel_data.get('errorMessage',
                                'Cancel failed')))
                        return self._convert_order_update(cancel_data[
                            'orderCancelTransaction'])
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
                async with session.get(URL(self.api_url).join(
                    f'/v3/accounts/{self.account_id}/orders')) as response:
                    data = await response.json()
                    order = next((o for o in data['orders'] if o.get(
                        'clientExtensions', {}).get('id') ==
                        client_order_id), None)
                    if not order:
                        return ExecutionReport(broker_order_id='',
                            client_order_id=client_order_id, instrument='',
                            status=OrderStatus.REJECTED, rejection_reason=
                            'Order not found')
                    request_data = {'order': {}}
                    if 'price' in modifications:
                        request_data['order']['price'] = str(modifications[
                            'price'])
                    if 'stop_loss' in modifications:
                        request_data['order']['stopLossOnFill'] = {'price':
                            str(modifications['stop_loss'])}
                    if 'take_profit' in modifications:
                        request_data['order']['takeProfitOnFill'] = {'price':
                            str(modifications['take_profit'])}
                    async with session.put(URL(self.api_url).join(
                        f"/v3/accounts/{self.account_id}/orders/{order['id']}"
                        ), json=request_data) as modify_response:
                        modify_data = await modify_response.json()
                        if modify_response.status != 200:
                            return ExecutionReport(broker_order_id=order[
                                'id'], client_order_id=client_order_id,
                                instrument=order['instrument'], status=
                                OrderStatus.REJECTED, rejection_reason=str(
                                modify_data.get('errorMessage',
                                'Modification failed')))
                        return self._convert_order_update(modify_data[
                            'orderCreateTransaction'])
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
                async with session.get(URL(self.api_url).join(
                    f'/v3/accounts/{self.account_id}/openPositions')
                    ) as response:
                    if response.status != 200:
                        return []
                    data = await response.json()
                    return [self._convert_position_update(pos) for pos in
                        data['positions']]
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
                async with session.get(URL(self.api_url).join(
                    f'/v3/accounts/{self.account_id}')) as response:
                    if response.status != 200:
                        raise Exception('Failed to get account information')
                    data = await response.json()
                    return self._convert_account_update(data['account'])
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
            instrument = position_id.split('_')[0]
            async with aiohttp.ClientSession(headers=self._get_auth_headers()
                ) as session:
                async with session.get(URL(self.api_url).join(
                    f'/v3/accounts/{self.account_id}/positions/{instrument}')
                    ) as response:
                    if response.status != 200:
                        return ExecutionReport(broker_order_id='',
                            client_order_id='', instrument=instrument,
                            status=OrderStatus.REJECTED, rejection_reason=
                            'Position not found')
                    position_data = await response.json()
                    position = position_data['position']
                    long_units = float(position.get('long', {}).get('units', 0)
                        )
                    short_units = float(position.get('short', {}).get(
                        'units', 0))
                    close_units = quantity if quantity else abs(long_units +
                        short_units)
                    request_data = {'longUnits': 'ALL' if not quantity else
                        str(close_units)} if long_units > 0 else {'shortUnits':
                        'ALL' if not quantity else str(close_units)}
                    async with session.put(URL(self.api_url).join(
                        f'/v3/accounts/{self.account_id}/positions/{instrument}/close'
                        ), json=request_data) as close_response:
                        close_data = await close_response.json()
                        if close_response.status != 200:
                            return ExecutionReport(broker_order_id='',
                                client_order_id='', instrument=instrument,
                                status=OrderStatus.REJECTED,
                                rejection_reason=str(close_data.get(
                                'errorMessage', 'Close failed')))
                        return self._convert_order_update(close_data[
                            'orderFillTransaction'])
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
            self._account_callback = callback_callback
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
        return 'OANDA'
