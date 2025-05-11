"""
Interactive Brokers adapter implementation.
"""
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import ibapi
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order as IBOrder
from ibapi.common import ListOfOrderState, BarData
from .base_broker_adapter import BaseBrokerAdapter
from ..interfaces.broker_adapter import OrderRequest, ExecutionReport, PositionUpdate, AccountUpdate, OrderType, OrderDirection, OrderStatus
logger = logging.getLogger(__name__)

class IBWrapper(EWrapper):
    """
    Wrapper class for Interactive Brokers callbacks.
    """

    def __init__(self):
        super().__init__()
        self._callbacks = {}
        self._account_updates = {}
        self._position_updates = {}
        self._order_updates = {}
        self._next_req_id = 1

    def register_callback(self, req_id: int, callback: callable) -> None:
        """Register a callback for a specific request ID."""
        self._callbacks[req_id] = callback

    def next_valid_id(self, orderId: int) -> None:
        """Callback for connection confirmation."""
        if 0 in self._callbacks:
            self._callbacks[0](True)

    def error(self, reqId: int, errorCode: int, errorString: str) -> None:
        """Handle error messages from IB."""
        logger.error(f'IB Error {errorCode}: {errorString} (reqId: {reqId})')
        if reqId in self._callbacks:
            self._callbacks[reqId](False, f'Error {errorCode}: {errorString}')

    def exec_details(self, reqId: int, contract: Contract, execution) -> None:
        """Handle execution reports."""
        if reqId in self._callbacks:
            self._callbacks[reqId]({'reqId': reqId, 'symbol': contract.symbol, 'orderId': execution.orderId, 'shares': execution.shares, 'price': execution.price, 'time': execution.time})

    def update_account_value(self, key: str, val: str, currency: str, accountName: str) -> None:
        """Handle account value updates."""
        if accountName not in self._account_updates:
            self._account_updates[accountName] = {}
        self._account_updates[accountName][key] = {'value': val, 'currency': currency}

    def update_portfolio(self, contract: Contract, position: float, marketPrice: float, marketValue: float, averageCost: float, unrealizedPNL: float, realizedPNL: float, accountName: str) -> None:
        """Handle portfolio/position updates."""
        key = f'{contract.symbol}_{contract.secType}_{contract.currency}'
        if accountName not in self._position_updates:
            self._position_updates[accountName] = {}
        self._position_updates[accountName][key] = {'position': position, 'marketPrice': marketPrice, 'marketValue': marketValue, 'averageCost': averageCost, 'unrealizedPNL': unrealizedPNL, 'realizedPNL': realizedPNL}

class IBClient(EClient):
    """
    Client class for Interactive Brokers connection.
    """

    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)
        self._lock = asyncio.Lock()

    async def run_async(self):
        """Run the client asynchronously."""
        while self.isConnected():
            await asyncio.sleep(0.1)
            self.run()

class InteractiveBrokersAdapter(BaseBrokerAdapter):
    """
    Interactive Brokers adapter implementation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the IB adapter.
        
        Args:
            config: Configuration dictionary containing:
                - host: TWS/Gateway host
                - port: TWS/Gateway port
                - client_id: Unique client ID
                Additional base adapter config parameters
        """
        super().__init__(config)
        self.host = config['host']
        self.port = config['port']
        self.client_id = config['client_id']
        self.wrapper = IBWrapper()
        self.client = IBClient(self.wrapper)
        self._order_map = {}
        self._position_map = {}
        self._req_id = 1

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        return {}

    async def connect(self) -> bool:
        """
        Establish connection to Interactive Brokers TWS/Gateway.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            future = asyncio.Future()
            self.wrapper.register_callback(0, future.set_result)
            self.client.connect(self.host, self.port, self.client_id)
            asyncio.create_task(self.client.run_async())
            result = await asyncio.wait_for(future, timeout=30)
            if not result:
                raise Exception('Connection failed')
            self._is_connected = True
            await self._start_heartbeat()
            self.client.reqAccountUpdates(True, '')
            logger.info('Successfully connected to Interactive Brokers')
            return True
        except Exception as e:
            logger.error(f'Failed to connect to Interactive Brokers: {str(e)}')
            return False

    async def disconnect(self) -> bool:
        """
        Disconnect from Interactive Brokers.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            self._is_connected = False
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            if self.client.isConnected():
                self.client.disconnect()
            logger.info('Successfully disconnected from Interactive Brokers')
            return True
        except Exception as e:
            logger.error(f'Error disconnecting from Interactive Brokers: {str(e)}')
            return False

    async def _send_heartbeat(self) -> None:
        """Verify connection health."""
        if not self.client.isConnected():
            raise Exception('Connection lost')

    def _create_contract(self, symbol: str) -> Contract:
        """Create an IB contract object."""
        contract = Contract()
        if '/' in symbol:
            base, quote = symbol.split('/')
            contract.symbol = base
            contract.currency = quote
            contract.secType = 'CASH'
            contract.exchange = 'IDEALPRO'
        else:
            contract.symbol = symbol
            contract.secType = 'STK'
            contract.currency = 'USD'
            contract.exchange = 'SMART'
        return contract

    def _create_ib_order(self, order_request: OrderRequest) -> IBOrder:
        """Create an IB order object."""
        ib_order = IBOrder()
        if order_request.order_type == OrderType.MARKET:
            ib_order.orderType = 'MKT'
        elif order_request.order_type == OrderType.LIMIT:
            ib_order.orderType = 'LMT'
            ib_order.lmtPrice = order_request.price
        elif order_request.order_type == OrderType.STOP:
            ib_order.orderType = 'STP'
            ib_order.auxPrice = order_request.price
        ib_order.action = 'BUY' if order_request.direction == OrderDirection.BUY else 'SELL'
        ib_order.totalQuantity = order_request.quantity
        if order_request.stop_loss:
            ib_order.orderType = 'STP'
            ib_order.auxPrice = order_request.stop_loss
        if order_request.take_profit:
            ib_order.orderType = 'LMT'
            ib_order.lmtPrice = order_request.take_profit
        return ib_order

    async def place_order(self, order_request: OrderRequest) -> ExecutionReport:
        """
        Place a new order with Interactive Brokers.
        
        Args:
            order_request: The order to place
            
        Returns:
            Execution report for the order
        """
        if not self._is_connected:
            self._order_queue.append(order_request)
            return ExecutionReport(broker_order_id='', client_order_id=order_request.client_order_id, instrument=order_request.instrument, status=OrderStatus.PENDING, rejection_reason='Not connected - order queued')
        try:
            contract = self._create_contract(order_request.instrument)
            ib_order = self._create_ib_order(order_request)
            self._order_map[order_request.client_order_id] = ib_order.orderId
            future = asyncio.Future()
            req_id = self._req_id
            self._req_id += 1
            self.wrapper.register_callback(req_id, future.set_result)
            self.client.placeOrder(ib_order.orderId, contract, ib_order)
            result = await asyncio.wait_for(future, timeout=30)
            if isinstance(result, bool) and (not result):
                return ExecutionReport(broker_order_id='', client_order_id=order_request.client_order_id, instrument=order_request.instrument, status=OrderStatus.REJECTED, rejection_reason='Order rejected by IB')
            return ExecutionReport(broker_order_id=str(ib_order.orderId), client_order_id=order_request.client_order_id, instrument=order_request.instrument, status=OrderStatus.FILLED if result['shares'] == order_request.quantity else OrderStatus.PARTIALLY_FILLED, filled_quantity=float(result['shares']), average_price=float(result['price']))
        except Exception as e:
            logger.error(f'Error placing order: {str(e)}')
            return ExecutionReport(broker_order_id='', client_order_id=order_request.client_order_id, instrument=order_request.instrument, status=OrderStatus.REJECTED, rejection_reason=str(e))

    async def cancel_order(self, client_order_id: str) -> ExecutionReport:
        """
        Cancel an existing order.
        
        Args:
            client_order_id: The client order ID to cancel
            
        Returns:
            Execution report indicating the cancellation status
        """
        try:
            ib_order_id = self._order_map.get(client_order_id)
            if not ib_order_id:
                return ExecutionReport(broker_order_id='', client_order_id=client_order_id, instrument='', status=OrderStatus.REJECTED, rejection_reason='Order not found')
            future = asyncio.Future()
            req_id = self._req_id
            self._req_id += 1
            self.wrapper.register_callback(req_id, future.set_result)
            self.client.cancelOrder(ib_order_id)
            result = await asyncio.wait_for(future, timeout=30)
            return ExecutionReport(broker_order_id=str(ib_order_id), client_order_id=client_order_id, instrument='', status=OrderStatus.CANCELLED if result else OrderStatus.REJECTED, rejection_reason='' if result else 'Cancel failed')
        except Exception as e:
            logger.error(f'Error cancelling order: {str(e)}')
            return ExecutionReport(broker_order_id='', client_order_id=client_order_id, instrument='', status=OrderStatus.REJECTED, rejection_reason=str(e))

    async def modify_order(self, client_order_id: str, modifications: Dict[str, Any]) -> ExecutionReport:
        """
        Modify an existing order.
        
        Args:
            client_order_id: The client order ID to modify
            modifications: Dictionary of fields to modify
            
        Returns:
            Execution report indicating the modification status
        """
        try:
            ib_order_id = self._order_map.get(client_order_id)
            if not ib_order_id:
                return ExecutionReport(broker_order_id='', client_order_id=client_order_id, instrument='', status=OrderStatus.REJECTED, rejection_reason='Order not found')
            future = asyncio.Future()
            req_id = self._req_id
            self._req_id += 1
            self.wrapper.register_callback(req_id, future.set_result)
            self.client.reqOpenOrders()
            orders = await asyncio.wait_for(future, timeout=30)
            target_order = None
            for order in orders:
                if order.orderId == ib_order_id:
                    target_order = order
                    break
            if not target_order:
                return ExecutionReport(broker_order_id=str(ib_order_id), client_order_id=client_order_id, instrument='', status=OrderStatus.REJECTED, rejection_reason='Order not found')
            if 'price' in modifications:
                target_order.lmtPrice = modifications['price']
            if 'stop_loss' in modifications:
                target_order.auxPrice = modifications['stop_loss']
            if 'quantity' in modifications:
                target_order.totalQuantity = modifications['quantity']
            future = asyncio.Future()
            req_id = self._req_id
            self._req_id += 1
            self.wrapper.register_callback(req_id, future.set_result)
            self.client.placeOrder(ib_order_id, target_order.contract, target_order)
            result = await asyncio.wait_for(future, timeout=30)
            return ExecutionReport(broker_order_id=str(ib_order_id), client_order_id=client_order_id, instrument=target_order.contract.symbol, status=OrderStatus.ACCEPTED if result else OrderStatus.REJECTED, rejection_reason='' if result else 'Modification failed')
        except Exception as e:
            logger.error(f'Error modifying order: {str(e)}')
            return ExecutionReport(broker_order_id='', client_order_id=client_order_id, instrument='', status=OrderStatus.REJECTED, rejection_reason=str(e))

    async def get_positions(self) -> List[PositionUpdate]:
        """
        Get all current positions.
        
        Returns:
            List of position updates
        """
        try:
            positions = []
            for account, pos_data in self.wrapper._position_updates.items():
                for symbol, details in pos_data.items():
                    positions.append(PositionUpdate(instrument=symbol.split('_')[0], position_id=f'{symbol}_{account}', quantity=details['position'], average_price=details['averageCost'], unrealized_pl=details['unrealizedPNL'], realized_pl=details['realizedPNL'], margin_used=0.0))
            return positions
        except Exception as e:
            logger.error(f'Error getting positions: {str(e)}')
            return []

    async def get_account_info(self) -> AccountUpdate:
        """
        Get current account information.
        
        Returns:
            Account update with current information
        """
        try:
            account = next(iter(self.wrapper._account_updates.keys()))
            account_data = self.wrapper._account_updates[account]
            return AccountUpdate(account_id=account, balance=float(account_data.get('NetLiquidation', {}).get('value', 0)), equity=float(account_data.get('EquityWithLoanValue', {}).get('value', 0)), margin_used=float(account_data.get('InitMarginReq', {}).get('value', 0)), margin_available=float(account_data.get('AvailableFunds', {}).get('value', 0)), currency=account_data.get('NetLiquidation', {}).get('currency', 'USD'))
        except Exception as e:
            logger.error(f'Error getting account info: {str(e)}')
            raise

    async def close_position(self, position_id: str, quantity: Optional[float]=None) -> ExecutionReport:
        """
        Close an existing position.
        
        Args:
            position_id: The position ID to close
            quantity: Optional quantity to close (if None, close entire position)
            
        Returns:
            Execution report for the closing order
        """
        try:
            symbol, account = position_id.split('_', 1)
            if account not in self.wrapper._position_updates:
                return ExecutionReport(broker_order_id='', client_order_id='', instrument=symbol, status=OrderStatus.REJECTED, rejection_reason='Position not found')
            pos_data = self.wrapper._position_updates[account].get(f'{symbol}_STK_USD')
            if not pos_data:
                return ExecutionReport(broker_order_id='', client_order_id='', instrument=symbol, status=OrderStatus.REJECTED, rejection_reason='Position not found')
            contract = self._create_contract(symbol)
            ib_order = IBOrder()
            ib_order.orderType = 'MKT'
            ib_order.action = 'SELL' if pos_data['position'] > 0 else 'BUY'
            ib_order.totalQuantity = quantity if quantity else abs(pos_data['position'])
            future = asyncio.Future()
            req_id = self._req_id
            self._req_id += 1
            self.wrapper.register_callback(req_id, future.set_result)
            self.client.placeOrder(ib_order.orderId, contract, ib_order)
            result = await asyncio.wait_for(future, timeout=30)
            return ExecutionReport(broker_order_id=str(ib_order.orderId), client_order_id='', instrument=symbol, status=OrderStatus.FILLED if result['shares'] == ib_order.totalQuantity else OrderStatus.PARTIALLY_FILLED, filled_quantity=float(result['shares']), average_price=float(result['price']))
        except Exception as e:
            logger.error(f'Error closing position: {str(e)}')
            return ExecutionReport(broker_order_id='', client_order_id='', instrument='', status=OrderStatus.REJECTED, rejection_reason=str(e))

    async def subscribe_to_updates(self, callback_execution: callable, callback_position: callable, callback_account: callable) -> bool:
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
            self.client.reqAccountUpdates(True, '')
            return True
        except Exception as e:
            logger.error(f'Error subscribing to updates: {str(e)}')
            return False

    async def unsubscribe_from_updates(self) -> bool:
        """
        Unsubscribe from real-time updates.
        
        Returns:
            True if unsubscription successful, False otherwise
        """
        try:
            self._execution_callback = None
            self._position_callback = None
            self._account_callback = None
            self.client.reqAccountUpdates(False, '')
            return True
        except Exception as e:
            logger.error(f'Error unsubscribing from updates: {str(e)}')
            return False

    @property
    def name(self) -> str:
        """Get the name of the broker."""
        return 'InteractiveBrokers'