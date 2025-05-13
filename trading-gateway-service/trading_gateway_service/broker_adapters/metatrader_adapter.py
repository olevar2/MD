"""
MetaTrader broker adapter implementation.
"""
import asyncio
import logging
import hashlib
import hmac
from datetime import datetime
from typing import Dict, List, Optional, Any
import MetaTrader5 as mt5
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

class MetaTraderAdapter(BaseBrokerAdapter):
    """
    MetaTrader 5 broker adapter implementation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MetaTrader adapter.
        
        Args:
            config: Configuration dictionary containing:
                - server: MetaTrader server name
                - login: MetaTrader account login
                - password: MetaTrader account password
                - timeout: Connection timeout in seconds
                Additional base adapter config parameters
        """
        super().__init__(config)
        self.server = config['server']
        self.login = config['login']
        self.password = config['password']
        self._symbol_map = {}
        self._position_map = {}

    def _get_auth_headers(self) ->Dict[str, str]:
        """Get authentication headers for API requests."""
        return {}

    @async_with_exception_handling
    async def connect(self) ->bool:
        """
        Establish connection to MetaTrader terminal.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if not mt5.initialize(login=self.login, server=self.server,
                password=self.password):
                logger.error(f'MT5 initialization failed: {mt5.last_error()}')
                return False
            self._is_connected = True
            await self._start_heartbeat()
            self._init_symbol_info()
            logger.info('Successfully connected to MetaTrader')
            return True
        except Exception as e:
            logger.error(f'Failed to connect to MetaTrader: {str(e)}')
            return False

    @async_with_exception_handling
    async def disconnect(self) ->bool:
        """
        Disconnect from MetaTrader terminal.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            self._is_connected = False
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            mt5.shutdown()
            logger.info('Successfully disconnected from MetaTrader')
            return True
        except Exception as e:
            logger.error(f'Error disconnecting from MetaTrader: {str(e)}')
            return False

    @with_exception_handling
    def _init_symbol_info(self) ->None:
        """Initialize symbol information and mappings."""
        try:
            symbols = mt5.symbols_get()
            for symbol in symbols:
                std_symbol = self._standardize_symbol(symbol.name)
                self._symbol_map[std_symbol] = symbol.name
        except Exception as e:
            logger.error(f'Error initializing symbol information: {str(e)}')

    def _standardize_symbol(self, mt5_symbol: str) ->str:
        """Convert MT5 symbol format to standard format (e.g., 'EURUSD')."""
        clean_symbol = mt5_symbol.strip().upper()
        return clean_symbol.replace('.', '')

    @async_with_exception_handling
    async def _send_heartbeat(self) ->None:
        """Send heartbeat to verify MT5 connection."""
        try:
            if not mt5.terminal_info():
                raise Exception('MT5 terminal not responding')
        except Exception as e:
            logger.error(f'Heartbeat failed: {str(e)}')
            raise

    @async_with_exception_handling
    async def place_order(self, order_request: OrderRequest) ->ExecutionReport:
        """
        Place a new order with MetaTrader.
        
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
            mt5_symbol = self._symbol_map.get(order_request.instrument)
            if not mt5_symbol:
                raise ValueError(f'Unknown symbol: {order_request.instrument}')
            request = {'action': mt5.TRADE_ACTION_DEAL if order_request.
                order_type == OrderType.MARKET else mt5.
                TRADE_ACTION_PENDING, 'symbol': mt5_symbol, 'volume':
                order_request.quantity, 'type': self._convert_order_type(
                order_request.order_type, order_request.direction), 'price':
                order_request.price or mt5.symbol_info_tick(mt5_symbol).ask,
                'deviation': self.config_manager.get('price_deviation', 10),
                'magic': 123456, 'comment':
                f'ID:{order_request.client_order_id}', 'type_time': mt5.
                ORDER_TIME_GTC}
            if order_request.stop_loss:
                request['sl'] = order_request.stop_loss
            if order_request.take_profit:
                request['tp'] = order_request.take_profit
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return ExecutionReport(broker_order_id='', client_order_id=
                    order_request.client_order_id, instrument=order_request
                    .instrument, status=OrderStatus.REJECTED,
                    rejection_reason=f'MT5 Error: {result.comment}')
            return ExecutionReport(broker_order_id=str(result.order),
                client_order_id=order_request.client_order_id, instrument=
                order_request.instrument, status=OrderStatus.FILLED if 
                result.volume == order_request.quantity else OrderStatus.
                PARTIALLY_FILLED, filled_quantity=result.volume,
                average_price=result.price)
        except Exception as e:
            logger.error(f'Error placing order: {str(e)}')
            return ExecutionReport(broker_order_id='', client_order_id=
                order_request.client_order_id, instrument=order_request.
                instrument, status=OrderStatus.REJECTED, rejection_reason=
                str(e))

    def _convert_order_type(self, order_type: OrderType, direction:
        OrderDirection) ->int:
        """Convert our order type to MT5 order type."""
        if order_type == OrderType.MARKET:
            return (mt5.ORDER_TYPE_BUY if direction == OrderDirection.BUY else
                mt5.ORDER_TYPE_SELL)
        elif order_type == OrderType.LIMIT:
            return (mt5.ORDER_TYPE_BUY_LIMIT if direction == OrderDirection
                .BUY else mt5.ORDER_TYPE_SELL_LIMIT)
        elif order_type == OrderType.STOP:
            return (mt5.ORDER_TYPE_BUY_STOP if direction == OrderDirection.
                BUY else mt5.ORDER_TYPE_SELL_STOP)
        raise ValueError(f'Unsupported order type: {order_type}')

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
            orders = mt5.orders_get()
            target_order = None
            for order in orders:
                if f'ID:{client_order_id}' in order.comment:
                    target_order = order
                    break
            if not target_order:
                return ExecutionReport(broker_order_id='', client_order_id=
                    client_order_id, instrument='', status=OrderStatus.
                    REJECTED, rejection_reason='Order not found')
            request = {'action': mt5.TRADE_ACTION_REMOVE, 'order':
                target_order.ticket}
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return ExecutionReport(broker_order_id=str(target_order.
                    ticket), client_order_id=client_order_id, instrument=
                    target_order.symbol, status=OrderStatus.REJECTED,
                    rejection_reason=f'MT5 Error: {result.comment}')
            return ExecutionReport(broker_order_id=str(target_order.ticket),
                client_order_id=client_order_id, instrument=target_order.
                symbol, status=OrderStatus.CANCELLED)
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
            orders = mt5.orders_get()
            target_order = None
            for order in orders:
                if f'ID:{client_order_id}' in order.comment:
                    target_order = order
                    break
            if not target_order:
                return ExecutionReport(broker_order_id='', client_order_id=
                    client_order_id, instrument='', status=OrderStatus.
                    REJECTED, rejection_reason='Order not found')
            request = {'action': mt5.TRADE_ACTION_MODIFY, 'order':
                target_order.ticket, 'price': modifications.get('price',
                target_order.price_open), 'sl': modifications.get(
                'stop_loss', target_order.sl), 'tp': modifications.get(
                'take_profit', target_order.tp)}
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return ExecutionReport(broker_order_id=str(target_order.
                    ticket), client_order_id=client_order_id, instrument=
                    target_order.symbol, status=OrderStatus.REJECTED,
                    rejection_reason=f'MT5 Error: {result.comment}')
            return ExecutionReport(broker_order_id=str(target_order.ticket),
                client_order_id=client_order_id, instrument=target_order.
                symbol, status=OrderStatus.ACCEPTED)
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
            positions = mt5.positions_get()
            if positions is None:
                return []
            result = []
            for pos in positions:
                position_id = str(pos.ticket)
                self._position_map[position_id] = pos.ticket
                result.append(PositionUpdate(instrument=self.
                    _standardize_symbol(pos.symbol), position_id=
                    position_id, quantity=pos.volume if pos.type == mt5.
                    POSITION_TYPE_BUY else -pos.volume, average_price=pos.
                    price_open, unrealized_pl=pos.profit, margin_used=pos.
                    margin))
            return result
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
            account_info = mt5.account_info()
            if account_info is None:
                raise Exception('Failed to get account information')
            return AccountUpdate(account_id=str(account_info.login),
                balance=account_info.balance, equity=account_info.equity,
                margin_used=account_info.margin, margin_available=
                account_info.margin_free, currency=account_info.currency)
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
            mt5_ticket = self._position_map.get(position_id)
            if not mt5_ticket:
                return ExecutionReport(broker_order_id='', client_order_id=
                    '', instrument='', status=OrderStatus.REJECTED,
                    rejection_reason='Position not found')
            position = mt5.positions_get(ticket=mt5_ticket)
            if not position:
                return ExecutionReport(broker_order_id='', client_order_id=
                    '', instrument='', status=OrderStatus.REJECTED,
                    rejection_reason='Position not found')
            position = position[0]
            close_volume = quantity if quantity else position.volume
            request = {'action': mt5.TRADE_ACTION_DEAL, 'position':
                mt5_ticket, 'symbol': position.symbol, 'volume':
                close_volume, 'type': mt5.ORDER_TYPE_SELL if position.type ==
                mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY, 'price': mt5
                .symbol_info_tick(position.symbol).bid if position.type ==
                mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(position.
                symbol).ask, 'deviation': self.config.get('price_deviation',
                10), 'magic': 123456, 'comment': f'Close:{position_id}'}
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return ExecutionReport(broker_order_id='', client_order_id=
                    '', instrument=position.symbol, status=OrderStatus.
                    REJECTED, rejection_reason=f'MT5 Error: {result.comment}')
            return ExecutionReport(broker_order_id=str(result.order),
                client_order_id='', instrument=position.symbol, status=
                OrderStatus.FILLED if result.volume == close_volume else
                OrderStatus.PARTIALLY_FILLED, filled_quantity=result.volume,
                average_price=result.price)
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
            asyncio.create_task(self._monitor_updates())
            return True
        except Exception as e:
            logger.error(f'Error subscribing to updates: {str(e)}')
            return False

    @async_with_exception_handling
    async def _monitor_updates(self) ->None:
        """Monitor for updates from MT5."""
        last_position_check = datetime.utcnow()
        while self._is_connected:
            try:
                trades = mt5.history_deals_get(from_date=last_position_check)
                if trades:
                    for trade in trades:
                        if self._execution_callback:
                            await self._execution_callback(self.
                                _convert_trade_to_execution(trade))
                now = datetime.utcnow()
                if (now - last_position_check).total_seconds() >= 5:
                    if self._position_callback:
                        positions = await self.get_positions()
                        for pos in positions:
                            await self._position_callback(pos)
                    if self._account_callback:
                        account_info = await self.get_account_info()
                        await self._account_callback(account_info)
                    last_position_check = now
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f'Error in update monitoring: {str(e)}')
                await asyncio.sleep(5)

    def _convert_trade_to_execution(self, trade) ->ExecutionReport:
        """Convert MT5 trade to ExecutionReport."""
        return ExecutionReport(broker_order_id=str(trade.order),
            client_order_id=self._extract_client_id(trade.comment),
            instrument=self._standardize_symbol(trade.symbol), status=
            OrderStatus.FILLED, filled_quantity=trade.volume, average_price
            =trade.price)

    def _extract_client_id(self, comment: str) ->str:
        """Extract client order ID from MT5 comment."""
        if comment and comment.startswith('ID:'):
            return comment[3:]
        return ''

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
        return 'MetaTrader5'
