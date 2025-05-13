"""
Forex Broker Simulator for Paper Trading.

Implements a realistic forex broker simulation including:
- Order execution with realistic slippage and partial fills
- Market data simulation with bid/ask spreads
- Position and account management
- Trading session handling
"""
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from decimal import Decimal
from enum import Enum
logger = logging.getLogger(__name__)


from trading_gateway_service.resilience.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class OrderType(str, Enum):
    """Order types supported by the broker simulator."""
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'


class OrderStatus(str, Enum):
    """Order status states."""
    PENDING = 'pending'
    ACCEPTED = 'accepted'
    PARTIALLY_FILLED = 'partially_filled'
    FILLED = 'filled'
    REJECTED = 'rejected'
    CANCELLED = 'cancelled'
    EXPIRED = 'expired'


class ExecutionType(str, Enum):
    """Types of order execution."""
    NEW = 'new'
    TRADE = 'trade'
    EXPIRED = 'expired'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'


class SimulatedBroker:
    """
    Simulated forex broker for paper trading.
    
    Implements realistic order execution, position management,
    and account handling with configurable market conditions.
    """

    def __init__(self, initial_balance: float=100000.0, base_currency: str=
        'USD', max_leverage: float=50.0, min_lot_size: float=0.01,
        spread_config: Optional[Dict[str, float]]=None, session_config:
        Optional[Dict[str, Any]]=None, slippage_model: Optional[Dict[str,
        Any]]=None):
    """
      init  .
    
    Args:
        initial_balance: Description of initial_balance
        base_currency: Description of base_currency
        max_leverage: Description of max_leverage
        min_lot_size: Description of min_lot_size
        spread_config: Description of spread_config
        float]]: Description of float]]
        session_config: Description of session_config
        Any]]: Description of Any]]
        slippage_model: Description of slippage_model
        Any]]: Description of Any]]
    
    """

        self.initial_balance = initial_balance
        self.base_currency = base_currency
        self.max_leverage = max_leverage
        self.min_lot_size = min_lot_size
        self.spread_config = spread_config or {'EUR/USD': 1.0, 'GBP/USD': 
            1.5, 'USD/JPY': 1.0, 'AUD/USD': 1.2, 'USD/CHF': 1.5, 'USD/CAD':
            1.5, 'NZD/USD': 1.5}
        self.session_config = session_config or {'sydney': {'open': '22:00',
            'close': '07:00'}, 'tokyo': {'open': '00:00', 'close': '09:00'},
            'london': {'open': '08:00', 'close': '17:00'}, 'new_york': {
            'open': '13:00', 'close': '22:00'}}
        self.slippage_model = slippage_model or {'base_slippage': 0.1,
            'volume_impact': 0.02, 'volatility_impact': 0.5}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.balance = initial_balance
        self.equity = initial_balance
        self.margin_used = 0.0
        self.order_id_counter = 0
        self.market_data: Dict[str, Dict[str, Any]] = {}
        self.current_session = None
        self.is_market_open = True

    def place_order(self, symbol: str, order_type: OrderType, direction:
        str, size: float, price: Optional[float]=None, stop_loss: Optional[
        float]=None, take_profit: Optional[float]=None, expiry: Optional[
        datetime]=None, **kwargs) ->Dict[str, Any]:
        """
        Place a new order.
        
        Args:
            symbol: Trading symbol
            order_type: Type of order
            direction: 'buy' or 'sell'
            size: Order size in lots
            price: Limit price for limit orders
            stop_loss: Stop loss price
            take_profit: Take profit price
            expiry: Order expiration time
            **kwargs: Additional order parameters
            
        Returns:
            Order details including ID and initial status
        """
        validation = self._validate_order(symbol, order_type, direction,
            size, price)
        if not validation['valid']:
            return {'order_id': None, 'status': OrderStatus.REJECTED,
                'reason': validation['reason']}
        self.order_id_counter += 1
        order_id = f'ORDER_{self.order_id_counter}'
        order = {'order_id': order_id, 'symbol': symbol, 'type': order_type,
            'direction': direction, 'size': size, 'price': price,
            'stop_loss': stop_loss, 'take_profit': take_profit, 'status':
            OrderStatus.PENDING, 'filled_size': 0.0, 'average_price': None,
            'expiry': expiry, 'submission_time': datetime.utcnow(),
            'last_update_time': datetime.utcnow(), 'executions': []}
        self.orders[order_id] = order
        if order_type == OrderType.MARKET:
            self._execute_market_order(order)
        return {'order_id': order_id, 'status': order['status'],
            'executions': order['executions']}

    def cancel_order(self, order_id: str) ->Dict[str, Any]:
        """Cancel a pending order."""
        if order_id not in self.orders:
            return {'success': False, 'reason': 'Order not found'}
        order = self.orders[order_id]
        if order['status'] not in [OrderStatus.PENDING, OrderStatus.
            PARTIALLY_FILLED]:
            return {'success': False, 'reason':
                f"Cannot cancel order in {order['status']} status"}
        order['status'] = OrderStatus.CANCELLED
        order['last_update_time'] = datetime.utcnow()
        return {'success': True, 'order_id': order_id, 'status': order[
            'status']}

    @with_market_data_resilience('update_market_data')
    def update_market_data(self, symbol: str, bid: float, ask: float,
        timestamp: datetime, volume: Optional[float]=None, volatility:
        Optional[float]=None) ->None:
        """Update market data for a symbol."""
        self.market_data[symbol] = {'bid': bid, 'ask': ask, 'timestamp':
            timestamp, 'volume': volume, 'volatility': volatility}
        self._process_pending_orders(symbol)
        if symbol in self.positions:
            self._update_position(symbol)

    @with_broker_api_resilience('get_account_summary')
    def get_account_summary(self) ->Dict[str, Any]:
        """Get current account status."""
        return {'balance': self.balance, 'equity': self.equity,
            'margin_used': self.margin_used, 'free_margin': self.equity -
            self.margin_used, 'margin_level': self.equity / self.
            margin_used * 100 if self.margin_used > 0 else 100, 'positions':
            len(self.positions), 'pending_orders': len([o for o in self.
            orders.values() if o['status'] == OrderStatus.PENDING])}

    def _validate_order(self, symbol: str, order_type: OrderType, direction:
        str, size: float, price: Optional[float]=None) ->Dict[str, Any]:
        """Validate order parameters."""
        result = {'valid': True, 'reason': None}
        if not self.is_market_open:
            result['valid'] = False
            result['reason'] = 'Market is closed'
            return result
        if symbol not in self.spread_config:
            result['valid'] = False
            result['reason'] = f'Invalid symbol: {symbol}'
            return result
        if size < self.min_lot_size:
            result['valid'] = False
            result['reason'] = f'Size below minimum: {self.min_lot_size}'
            return result
        if direction not in ['buy', 'sell']:
            result['valid'] = False
            result['reason'] = f'Invalid direction: {direction}'
            return result
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT
            ] and price is None:
            result['valid'] = False
            result['reason'] = 'Price required for limit orders'
            return result
        required_margin = self._calculate_required_margin(symbol, size)
        if required_margin > self.equity - self.margin_used:
            result['valid'] = False
            result['reason'] = 'Insufficient margin'
            return result
        return result

    def _execute_market_order(self, order: Dict[str, Any]) ->None:
        """Execute a market order."""
        symbol = order['symbol']
        if symbol not in self.market_data:
            order['status'] = OrderStatus.REJECTED
            order['last_update_time'] = datetime.utcnow()
            return
        price = self.market_data[symbol]['ask'] if order['direction'
            ] == 'buy' else self.market_data[symbol]['bid']
        execution_price = self._apply_slippage(symbol, price, order[
            'direction'], order['size'])
        execution = {'time': datetime.utcnow(), 'type': ExecutionType.TRADE,
            'price': execution_price, 'size': order['size']}
        order['executions'].append(execution)
        order['status'] = OrderStatus.FILLED
        order['filled_size'] = order['size']
        order['average_price'] = execution_price
        order['last_update_time'] = datetime.utcnow()
        self._update_position_after_execution(order, execution)

    def _process_pending_orders(self, symbol: str) ->None:
        """Process pending orders for a symbol."""
        if symbol not in self.market_data:
            return
        current_bid = self.market_data[symbol]['bid']
        current_ask = self.market_data[symbol]['ask']
        for order in self.orders.values():
            if order['symbol'] != symbol or order['status'
                ] != OrderStatus.PENDING:
                continue
            if order['type'] == OrderType.LIMIT:
                self._check_limit_order(order, current_bid, current_ask)
            elif order['type'] == OrderType.STOP:
                self._check_stop_order(order, current_bid, current_ask)

    def _check_limit_order(self, order: Dict[str, Any], current_bid: float,
        current_ask: float) ->None:
        """Check if a limit order should be executed."""
        if order['direction'] == 'buy' and current_ask <= order['price']:
            self._execute_market_order(order)
        elif order['direction'] == 'sell' and current_bid >= order['price']:
            self._execute_market_order(order)

    def _check_stop_order(self, order: Dict[str, Any], current_bid: float,
        current_ask: float) ->None:
        """Check if a stop order should be executed."""
        if order['direction'] == 'buy' and current_ask >= order['price']:
            self._execute_market_order(order)
        elif order['direction'] == 'sell' and current_bid <= order['price']:
            self._execute_market_order(order)

    def _update_position(self, symbol: str) ->None:
        """Update position P&L and check for stop/limit triggers."""
        if symbol not in self.positions:
            return
        position = self.positions[symbol]
        current_bid = self.market_data[symbol]['bid']
        current_ask = self.market_data[symbol]['ask']
        if position['direction'] == 'buy':
            position['unrealized_pnl'] = (current_bid - position[
                'average_price']) * position['size']
        else:
            position['unrealized_pnl'] = (position['average_price'] -
                current_ask) * position['size']
        if position['stop_loss']:
            if position['direction'] == 'buy' and current_bid <= position[
                'stop_loss']:
                self._close_position(symbol, current_bid)
            elif position['direction'] == 'sell' and current_ask >= position[
                'stop_loss']:
                self._close_position(symbol, current_ask)
        if position['take_profit']:
            if position['direction'] == 'buy' and current_bid >= position[
                'take_profit']:
                self._close_position(symbol, current_bid)
            elif position['direction'] == 'sell' and current_ask <= position[
                'take_profit']:
                self._close_position(symbol, current_ask)
        self._update_account_metrics()

    def _update_position_after_execution(self, order: Dict[str, Any],
        execution: Dict[str, Any]) ->None:
        """Update position after order execution."""
        symbol = order['symbol']
        if symbol not in self.positions:
            self.positions[symbol] = {'symbol': symbol, 'direction': order[
                'direction'], 'size': execution['size'], 'average_price':
                execution['price'], 'unrealized_pnl': 0.0, 'stop_loss':
                order['stop_loss'], 'take_profit': order['take_profit']}
        else:
            position = self.positions[symbol]
            if position['direction'] == order['direction']:
                total_size = position['size'] + execution['size']
                position['average_price'] = (position['average_price'] *
                    position['size'] + execution['price'] * execution['size']
                    ) / total_size
                position['size'] = total_size
            elif execution['size'] >= position['size']:
                self._close_position(symbol, execution['price'])
            else:
                position['size'] -= execution['size']
        self._update_account_metrics()

    def _close_position(self, symbol: str, price: float) ->None:
        """Close a position."""
        if symbol not in self.positions:
            return
        position = self.positions[symbol]
        if position['direction'] == 'buy':
            pnl = (price - position['average_price']) * position['size']
        else:
            pnl = (position['average_price'] - price) * position['size']
        self.balance += pnl
        del self.positions[symbol]
        self._update_account_metrics()

    def _update_account_metrics(self) ->None:
        """Update account equity and margin metrics."""
        unrealized_pnl = sum(p['unrealized_pnl'] for p in self.positions.
            values())
        self.equity = self.balance + unrealized_pnl
        self.margin_used = sum(self._calculate_required_margin(p['symbol'],
            p['size']) for p in self.positions.values())

    def _calculate_required_margin(self, symbol: str, size: float) ->float:
        """Calculate required margin for a position."""
        if symbol not in self.market_data:
            return float('inf')
        price = self.market_data[symbol]['ask']
        margin_rate = 1 / self.max_leverage
        return price * size * margin_rate

    def _apply_slippage(self, symbol: str, base_price: float, direction:
        str, size: float) ->float:
        """Apply realistic slippage to execution price."""
        slippage_pips = self.slippage_model['base_slippage']
        volume_pips = size / 1000000 * self.slippage_model['volume_impact']
        slippage_pips += volume_pips
        if symbol in self.market_data and self.market_data[symbol].get(
            'volatility'):
            volatility = self.market_data[symbol]['volatility']
            volatility_pips = volatility * self.slippage_model[
                'volatility_impact']
            slippage_pips += volatility_pips
        pip_value = 0.0001 if 'JPY' not in symbol else 0.01
        slippage = slippage_pips * pip_value
        if direction == 'buy':
            return base_price + slippage
        else:
            return base_price - slippage
