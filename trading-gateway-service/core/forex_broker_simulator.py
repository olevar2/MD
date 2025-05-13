"""
ForexBrokerSimulator

This module provides an enhanced simulation of a forex broker, complete with
realistic market conditions, order execution, slippage models, and liquidity simulation.
It supports various market regime scenarios and can simulate extreme events.
"""
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import random
import json
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from utils.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class MarketRegimeType(Enum):
    """
    MarketRegimeType class that inherits from Enum.
    
    Attributes:
        Add attributes here
    """

    TRENDING = 'trending'
    RANGING = 'ranging'
    VOLATILE = 'volatile'
    BREAKOUT = 'breakout'
    CRISIS = 'crisis'
    NORMAL = 'normal'


class LiquidityLevel(Enum):
    """
    LiquidityLevel class that inherits from Enum.
    
    Attributes:
        Add attributes here
    """

    VERY_LOW = 'very_low'
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    VERY_HIGH = 'very_high'


class MarketEventType(Enum):
    """
    MarketEventType class that inherits from Enum.
    
    Attributes:
        Add attributes here
    """

    ECONOMIC_RELEASE = 'economic_release'
    CENTRAL_BANK_DECISION = 'central_bank_decision'
    GEOPOLITICAL_EVENT = 'geopolitical_event'
    FLASH_CRASH = 'flash_crash'
    LIQUIDITY_GAP = 'liquidity_gap'
    TECHNICAL_BREAKOUT = 'technical_breakout'


class OrderType(Enum):
    """
    OrderType class that inherits from Enum.
    
    Attributes:
        Add attributes here
    """

    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'


class OrderStatus(Enum):
    """
    OrderStatus class that inherits from Enum.
    
    Attributes:
        Add attributes here
    """

    PENDING = 'pending'
    FILLED = 'filled'
    PARTIALLY_FILLED = 'partially_filled'
    CANCELED = 'canceled'
    REJECTED = 'rejected'
    EXPIRED = 'expired'


class OrderSide(Enum):
    BUY = 'buy'
    SELL = 'sell'


class Order:
    """
    Represents a forex order with all relevant details.
    """

    def __init__(self, symbol: str, side: OrderSide, order_type: OrderType,
        quantity: float, price: Optional[float]=None, stop_price: Optional[
        float]=None, limit_price: Optional[float]=None, time_in_force: str=
        'GTC', order_id: Optional[str]=None, status: OrderStatus=
        OrderStatus.PENDING, created_at: Optional[datetime]=None, filled_at:
        Optional[datetime]=None, filled_price: Optional[float]=None,
        filled_quantity: float=0.0, client_order_id: Optional[str]=None):
    """
      init  .
    
    Args:
        symbol: Description of symbol
        side: Description of side
        order_type: Description of order_type
        quantity: Description of quantity
        price: Description of price
        stop_price: Description of stop_price
        limit_price: Description of limit_price
        time_in_force: Description of time_in_force
        order_id: Description of order_id
        status: Description of status
        created_at: Description of created_at
        filled_at: Description of filled_at
        filled_price: Description of filled_price
        filled_quantity: Description of filled_quantity
        client_order_id: Description of client_order_id
    
    """

        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.limit_price = limit_price
        self.time_in_force = time_in_force
        self.order_id = (order_id or
            f'ord_{int(datetime.now().timestamp() * 1000)}')
        self.status = status
        self.created_at = created_at or datetime.now()
        self.filled_at = filled_at
        self.filled_price = filled_price
        self.filled_quantity = filled_quantity
        self.client_order_id = client_order_id

    def to_dict(self) ->Dict[str, Any]:
        """Convert the order to a dictionary representation."""
        return {'order_id': self.order_id, 'client_order_id': self.
            client_order_id, 'symbol': self.symbol, 'side': self.side.value,
            'order_type': self.order_type.value, 'quantity': self.quantity,
            'price': self.price, 'stop_price': self.stop_price,
            'limit_price': self.limit_price, 'time_in_force': self.
            time_in_force, 'status': self.status.value, 'created_at': self.
            created_at.isoformat() if self.created_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else
            None, 'filled_price': self.filled_price, 'filled_quantity':
            self.filled_quantity}


class MarketConditionConfig:
    """
    Configuration for market conditions in the simulation.
    """

    def __init__(self, regime: MarketRegimeType=MarketRegimeType.NORMAL,
        volatility_factor: float=1.0, liquidity_level: LiquidityLevel=
        LiquidityLevel.MEDIUM, spread_factor: float=1.0, event_frequency:
        float=0.05, max_slippage: float=0.0003, gap_probability: float=0.01,
        session_volume_profile: Dict[str, float]=None):
    """
      init  .
    
    Args:
        regime: Description of regime
        volatility_factor: Description of volatility_factor
        liquidity_level: Description of liquidity_level
        spread_factor: Description of spread_factor
        event_frequency: Description of event_frequency
        max_slippage: Description of max_slippage
        gap_probability: Description of gap_probability
        session_volume_profile: Description of session_volume_profile
        float]: Description of float]
    
    """

        self.regime = regime
        self.volatility_factor = volatility_factor
        self.liquidity_level = liquidity_level
        self.spread_factor = spread_factor
        self.event_frequency = event_frequency
        self.max_slippage = max_slippage
        self.gap_probability = gap_probability
        self.session_volume_profile = session_volume_profile or {'sydney': 
            0.1, 'tokyo': 0.2, 'london': 0.4, 'new_york': 0.3}

    @classmethod
    def create_for_regime(cls, regime: MarketRegimeType
        ) ->'MarketConditionConfig':
        """
        Create a preset configuration for a specific market regime.
        
        Args:
            regime: The market regime type
            
        Returns:
            A MarketConditionConfig instance with settings appropriate for the regime
        """
        if regime == MarketRegimeType.TRENDING:
            return cls(regime=regime, volatility_factor=0.8,
                liquidity_level=LiquidityLevel.HIGH, spread_factor=0.9,
                event_frequency=0.03, gap_probability=0.005)
        elif regime == MarketRegimeType.RANGING:
            return cls(regime=regime, volatility_factor=0.7,
                liquidity_level=LiquidityLevel.MEDIUM, spread_factor=1.0,
                event_frequency=0.02, gap_probability=0.002)
        elif regime == MarketRegimeType.VOLATILE:
            return cls(regime=regime, volatility_factor=2.0,
                liquidity_level=LiquidityLevel.MEDIUM, spread_factor=1.5,
                event_frequency=0.1, gap_probability=0.03)
        elif regime == MarketRegimeType.BREAKOUT:
            return cls(regime=regime, volatility_factor=1.5,
                liquidity_level=LiquidityLevel.LOW, spread_factor=1.3,
                event_frequency=0.05, gap_probability=0.04)
        elif regime == MarketRegimeType.CRISIS:
            return cls(regime=regime, volatility_factor=3.0,
                liquidity_level=LiquidityLevel.VERY_LOW, spread_factor=3.0,
                event_frequency=0.2, gap_probability=0.08,
                session_volume_profile={'sydney': 0.05, 'tokyo': 0.15,
                'london': 0.5, 'new_york': 0.3})
        else:
            return cls(regime=MarketRegimeType.NORMAL, volatility_factor=
                1.0, liquidity_level=LiquidityLevel.MEDIUM, spread_factor=
                1.0, event_frequency=0.05, gap_probability=0.01)


class OrderBookLevel:
    """
    Represents a price level in the order book with volume.
    """

    def __init__(self, price: float, volume: float):
        self.price = price
        self.volume = volume


class OrderBook:
    """
    Simulated order book with bid and ask levels.
    """

    def __init__(self, mid_price: float, spread_pips: float, num_levels:
        int=10, liquidity_profile: LiquidityLevel=LiquidityLevel.MEDIUM):
    """
      init  .
    
    Args:
        mid_price: Description of mid_price
        spread_pips: Description of spread_pips
        num_levels: Description of num_levels
        liquidity_profile: Description of liquidity_profile
    
    """

        self.mid_price = mid_price
        self.spread_pips = spread_pips
        self.num_levels = num_levels
        self.liquidity_profile = liquidity_profile
        self.bid_levels: List[OrderBookLevel] = []
        self.ask_levels: List[OrderBookLevel] = []
        self._generate_order_book()

    def _generate_order_book(self):
        """
        Generate a realistic order book based on current price and liquidity profile.
        """
        pip_size = 0.0001
        spread_in_price = self.spread_pips * pip_size
        bid_price = self.mid_price - spread_in_price / 2
        ask_price = self.mid_price + spread_in_price / 2
        if self.liquidity_profile == LiquidityLevel.VERY_LOW:
            base_volume = 0.5
            volume_decay = 2.0
        elif self.liquidity_profile == LiquidityLevel.LOW:
            base_volume = 1.0
            volume_decay = 1.8
        elif self.liquidity_profile == LiquidityLevel.MEDIUM:
            base_volume = 2.0
            volume_decay = 1.5
        elif self.liquidity_profile == LiquidityLevel.HIGH:
            base_volume = 5.0
            volume_decay = 1.3
        else:
            base_volume = 10.0
            volume_decay = 1.2
        self.bid_levels = []
        for i in range(self.num_levels):
            level_price = bid_price - i * pip_size
            level_volume = base_volume * volume_decay ** i * (0.8 + 0.4 *
                random.random())
            self.bid_levels.append(OrderBookLevel(level_price, level_volume))
        self.ask_levels = []
        for i in range(self.num_levels):
            level_price = ask_price + i * pip_size
            level_volume = base_volume * volume_decay ** i * (0.8 + 0.4 *
                random.random())
            self.ask_levels.append(OrderBookLevel(level_price, level_volume))

    @with_market_data_resilience('update_mid_price')
    def update_mid_price(self, new_mid_price: float):
        """
        Update the order book with a new mid price.
        
        Args:
            new_mid_price: The new mid price
        """
        self.mid_price = new_mid_price
        self._generate_order_book()

    @with_broker_api_resilience('update_spread')
    def update_spread(self, new_spread_pips: float):
        """
        Update the order book with a new spread.
        
        Args:
            new_spread_pips: The new spread in pips
        """
        self.spread_pips = new_spread_pips
        self._generate_order_book()

    @with_broker_api_resilience('get_best_bid')
    def get_best_bid(self) ->float:
        """Get the current best bid price."""
        return self.bid_levels[0].price if self.bid_levels else 0

    @with_broker_api_resilience('get_best_ask')
    def get_best_ask(self) ->float:
        """Get the current best ask price."""
        return self.ask_levels[0].price if self.ask_levels else 0

    @with_broker_api_resilience('get_current_spread')
    def get_current_spread(self) ->float:
        """Get the current spread in price terms."""
        return self.get_best_ask() - self.get_best_bid()


class MarketEvent:
    """
    Represents a market event that affects price, volatility, and liquidity.
    """

    def __init__(self, event_type: MarketEventType, impact_magnitude: float,
        duration_minutes: int, description: str, affected_symbols: List[str]):
    """
      init  .
    
    Args:
        event_type: Description of event_type
        impact_magnitude: Description of impact_magnitude
        duration_minutes: Description of duration_minutes
        description: Description of description
        affected_symbols: Description of affected_symbols
    
    """

        self.event_type = event_type
        self.impact_magnitude = impact_magnitude
        self.duration_minutes = duration_minutes
        self.description = description
        self.affected_symbols = affected_symbols
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(minutes=duration_minutes)

    def is_active(self, current_time: datetime) ->bool:
        """Check if the event is currently active."""
        return self.start_time <= current_time <= self.end_time

    @with_market_data_resilience('get_price_impact_factor')
    def get_price_impact_factor(self) ->float:
        """Get the impact factor on price."""
        if self.event_type == MarketEventType.FLASH_CRASH:
            return -0.1 * self.impact_magnitude
        elif self.event_type == MarketEventType.LIQUIDITY_GAP:
            return 0.05 * self.impact_magnitude * (1 if random.random() > 
                0.5 else -1)
        elif self.event_type == MarketEventType.TECHNICAL_BREAKOUT:
            return 0.03 * self.impact_magnitude * (1 if random.random() > 
                0.5 else -1)
        elif self.event_type == MarketEventType.CENTRAL_BANK_DECISION:
            return 0.02 * self.impact_magnitude * (1 if random.random() > 
                0.5 else -1)
        elif self.event_type == MarketEventType.ECONOMIC_RELEASE:
            return 0.01 * self.impact_magnitude * (1 if random.random() > 
                0.5 else -1)
        else:
            return 0.005 * self.impact_magnitude * (1 if random.random() > 
                0.5 else -1)

    @with_broker_api_resilience('get_volatility_impact_factor')
    def get_volatility_impact_factor(self) ->float:
        """Get the impact factor on volatility."""
        if self.event_type == MarketEventType.FLASH_CRASH:
            return 3.0 * self.impact_magnitude
        elif self.event_type == MarketEventType.CENTRAL_BANK_DECISION:
            return 2.0 * self.impact_magnitude
        elif self.event_type == MarketEventType.ECONOMIC_RELEASE:
            return 1.5 * self.impact_magnitude
        else:
            return 1.0 * self.impact_magnitude

    @with_broker_api_resilience('get_spread_impact_factor')
    def get_spread_impact_factor(self) ->float:
        """Get the impact factor on spread."""
        if self.event_type == MarketEventType.FLASH_CRASH:
            return 5.0 * self.impact_magnitude
        elif self.event_type == MarketEventType.LIQUIDITY_GAP:
            return 3.0 * self.impact_magnitude
        elif self.event_type == MarketEventType.CENTRAL_BANK_DECISION:
            return 2.0 * self.impact_magnitude
        else:
            return 1.0 * self.impact_magnitude

    @with_broker_api_resilience('get_liquidity_impact')
    def get_liquidity_impact(self) ->LiquidityLevel:
        """Get the impact on liquidity levels."""
        if self.impact_magnitude > 0.8:
            return LiquidityLevel.VERY_LOW
        elif self.impact_magnitude > 0.6:
            return LiquidityLevel.LOW
        elif self.impact_magnitude > 0.3:
            return LiquidityLevel.MEDIUM
        else:
            return LiquidityLevel.HIGH


class ForexBrokerSimulator:
    """
    Advanced forex broker simulator that provides realistic market behavior,
    order execution, and supports different market regimes.
    """

    def __init__(self, initial_prices: Dict[str, float]=None, base_spreads:
        Dict[str, float]=None, initial_condition: MarketConditionConfig=
        None, balance: float=10000.0, leverage: float=100.0, data_source:
        Optional[pd.DataFrame]=None, replay_mode: bool=False,
        commission_per_lot: float=7.0, time_acceleration: float=1.0):
        """
        Initialize the forex broker simulator.
        
        Args:
            initial_prices: Dictionary of {symbol: price}
            base_spreads: Dictionary of {symbol: spread_in_pips}
            initial_condition: Initial market condition configuration
            balance: Initial account balance
            leverage: Account leverage
            data_source: Historical price data for replay mode
            replay_mode: Whether to use historical data replay
            commission_per_lot: Commission per standard lot (100k units)
            time_acceleration: Speed multiplier for time passage
        """
        self.initial_prices = initial_prices or {'EUR/USD': 1.1, 'GBP/USD':
            1.3, 'USD/JPY': 110.0, 'AUD/USD': 0.7, 'USD/CAD': 1.3}
        self.prices = self.initial_prices.copy()
        self.base_spreads = base_spreads or {'EUR/USD': 1.0, 'GBP/USD': 1.5,
            'USD/JPY': 1.0, 'AUD/USD': 1.2, 'USD/CAD': 1.5}
        self.current_spreads = {s: spread for s, spread in self.
            base_spreads.items()}
        self.market_condition = initial_condition or MarketConditionConfig()
        self.balance = balance
        self.initial_balance = balance
        self.leverage = leverage
        self.data_source = data_source
        self.replay_mode = replay_mode
        self.commission_per_lot = commission_per_lot
        self.time_acceleration = time_acceleration
        self.current_time = datetime.now()
        self.start_time = self.current_time
        self.simulation_time_elapsed = timedelta(0)
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Dict[str, float]] = {}
        self.order_books: Dict[str, OrderBook] = {}
        self.active_events: List[MarketEvent] = []
        self.transactions: List[Dict[str, Any]] = []
        self._initialize_order_books()
        logger.info(
            f'ForexBrokerSimulator initialized with {len(self.prices)} symbols'
            )

    def _initialize_order_books(self):
        """Initialize order books for all symbols."""
        for symbol, price in self.prices.items():
            spread = self.current_spreads[symbol]
            liquidity = self.market_condition.liquidity_level
            self.order_books[symbol] = OrderBook(price, spread,
                liquidity_profile=liquidity)

    def advance_time(self, seconds: float) ->None:
        """
        Advance the simulation time by the specified number of seconds.
        
        Args:
            seconds: Number of seconds to advance (will be multiplied by time_acceleration)
        """
        actual_seconds = seconds * self.time_acceleration
        self.current_time += timedelta(seconds=actual_seconds)
        self.simulation_time_elapsed += timedelta(seconds=actual_seconds)
        if not self.replay_mode:
            self._update_prices(actual_seconds)
        self._process_pending_orders()
        self._update_events()
        if random.random() < self.market_condition.event_frequency * (
            actual_seconds / 3600):
            self._generate_market_event()

    def _update_prices(self, seconds_elapsed: float):
        """
        Update prices based on time elapsed and current market conditions.
        
        Args:
            seconds_elapsed: Time elapsed in seconds
        """
        base_hourly_volatility = {'EUR/USD': 0.001, 'GBP/USD': 0.0015,
            'USD/JPY': 0.0012, 'AUD/USD': 0.0018, 'USD/CAD': 0.0014}
        for symbol, price in self.prices.items():
            hourly_vol = base_hourly_volatility.get(symbol, 0.0015)
            vol_scaling = self.market_condition.volatility_factor
            for event in self.active_events:
                if symbol in event.affected_symbols:
                    vol_scaling *= event.get_volatility_impact_factor()
            second_vol = hourly_vol * vol_scaling / np.sqrt(3600)
            drift = 0
            diffusion = np.random.normal(0, second_vol) * price
            event_impact = 0
            for event in self.active_events:
                if symbol in event.affected_symbols:
                    event_impact += price * event.get_price_impact_factor()
            new_price = price + drift + diffusion + event_impact
            self.prices[symbol] = max(0.0001, new_price)
            base_spread = self.base_spreads[symbol]
            spread_factor = self.market_condition.spread_factor
            for event in self.active_events:
                if symbol in event.affected_symbols:
                    spread_factor *= event.get_spread_impact_factor()
            self.current_spreads[symbol] = base_spread * spread_factor
            self.order_books[symbol].update_mid_price(self.prices[symbol])
            self.order_books[symbol].update_spread(self.current_spreads[symbol]
                )
            gap_probability = self.market_condition.gap_probability * (
                seconds_elapsed / 3600)
            if random.random() < gap_probability:
                gap_size = random.uniform(0.0005, 0.003) * price
                direction = 1 if random.random() > 0.5 else -1
                gap_price = price + gap_size * direction
                self.prices[symbol] = max(0.0001, gap_price)
                self.order_books[symbol].update_mid_price(self.prices[symbol])

    def _update_events(self):
        """Update and expire market events."""
        active_events = []
        for event in self.active_events:
            if event.is_active(self.current_time):
                active_events.append(event)
            else:
                logger.info(f'Market event expired: {event.description}')
        self.active_events = active_events

    def _generate_market_event(self):
        """Generate a random market event."""
        event_types = list(MarketEventType)
        weights = [0.4, 0.2, 0.1, 0.05, 0.1, 0.15]
        event_type = random.choices(event_types, weights=weights, k=1)[0]
        if event_type == MarketEventType.FLASH_CRASH:
            impact = random.uniform(0.7, 1.0)
            duration = random.randint(5, 30)
        elif event_type == MarketEventType.LIQUIDITY_GAP:
            impact = random.uniform(0.5, 0.9)
            duration = random.randint(5, 60)
        elif event_type == MarketEventType.CENTRAL_BANK_DECISION:
            impact = random.uniform(0.4, 0.8)
            duration = random.randint(60, 240)
        elif event_type == MarketEventType.ECONOMIC_RELEASE:
            impact = random.uniform(0.2, 0.7)
            duration = random.randint(15, 90)
        else:
            impact = random.uniform(0.1, 0.5)
            duration = random.randint(30, 180)
        all_symbols = list(self.prices.keys())
        num_affected = random.randint(1, max(1, len(all_symbols) // 2))
        affected_symbols = random.sample(all_symbols, num_affected)
        descriptions = {MarketEventType.FLASH_CRASH:
            'Sudden market flash crash', MarketEventType.LIQUIDITY_GAP:
            'Significant liquidity gap', MarketEventType.
            CENTRAL_BANK_DECISION: 'Central bank rate decision',
            MarketEventType.ECONOMIC_RELEASE: 'Major economic data release',
            MarketEventType.GEOPOLITICAL_EVENT:
            'Breaking geopolitical development', MarketEventType.
            TECHNICAL_BREAKOUT: 'Technical level breakout'}
        description = descriptions[event_type
            ] + f" affecting {', '.join(affected_symbols)}"
        event = MarketEvent(event_type=event_type, impact_magnitude=impact,
            duration_minutes=duration, description=description,
            affected_symbols=affected_symbols)
        self.active_events.append(event)
        logger.info(
            f'Generated market event: {description}, impact: {impact:.2f}, duration: {duration} minutes'
            )
        return event

    def _process_pending_orders(self):
        """Process all pending orders based on current prices."""
        for order_id, order in list(self.orders.items()):
            if order.status == OrderStatus.PENDING:
                if order.order_type == OrderType.MARKET:
                    self._execute_market_order(order)
                elif order.order_type == OrderType.LIMIT:
                    self._check_limit_order(order)
                elif order.order_type == OrderType.STOP:
                    self._check_stop_order(order)
                elif order.order_type == OrderType.STOP_LIMIT:
                    self._check_stop_limit_order(order)

    def _execute_market_order(self, order: Order) ->bool:
        """
        Execute a market order with realistic slippage and partial fills.
        
        Args:
            order: The order to execute
            
        Returns:
            True if execution was successful
        """
        symbol = order.symbol
        order_book = self.order_books.get(symbol)
        if not order_book:
            logger.error(f'No order book for symbol {symbol}')
            order.status = OrderStatus.REJECTED
            return False
        if order.side == OrderSide.BUY:
            base_price = order_book.get_best_ask()
            slippage = self._calculate_slippage(order, is_buy=True)
            execution_price = base_price * (1 + slippage)
        else:
            base_price = order_book.get_best_bid()
            slippage = self._calculate_slippage(order, is_buy=False)
            execution_price = base_price * (1 - slippage)
        fill_quantity = order.quantity
        is_partial = False
        if self.market_condition.liquidity_level in [LiquidityLevel.
            VERY_LOW, LiquidityLevel.LOW] and order.quantity > 5.0:
            fill_ratio = random.uniform(0.6, 0.95)
            fill_quantity = order.quantity * fill_ratio
            is_partial = True
        if is_partial:
            order.status = OrderStatus.PARTIALLY_FILLED
            order.filled_quantity = fill_quantity
        else:
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.filled_at = self.current_time
        self._update_position(order)
        self._record_transaction(order)
        return True

    def _calculate_slippage(self, order: Order, is_buy: bool) ->float:
        """
        Calculate realistic slippage based on order size, market conditions, and events.
        
        Args:
            order: The order
            is_buy: Whether this is a buy order
            
        Returns:
            Slippage as a decimal percentage
        """
        symbol = order.symbol
        base_slippage = {LiquidityLevel.VERY_LOW: 0.0005, LiquidityLevel.
            LOW: 0.0003, LiquidityLevel.MEDIUM: 0.0001, LiquidityLevel.HIGH:
            5e-05, LiquidityLevel.VERY_HIGH: 2e-05}
        slippage = base_slippage.get(self.market_condition.liquidity_level,
            0.0001)
        size_factor = min(5.0, 1.0 + order.quantity / 10.0)
        slippage *= size_factor
        for event in self.active_events:
            if symbol in event.affected_symbols:
                slippage *= 1.0 + event.get_volatility_impact_factor() / 2
        return min(slippage, self.market_condition.max_slippage)

    def _check_limit_order(self, order: Order):
        """Check if a limit order should be executed based on price."""
        symbol = order.symbol
        order_book = self.order_books.get(symbol)
        if not order_book:
            return
        if order.side == OrderSide.BUY:
            if order_book.get_best_ask() <= order.price:
                self._execute_market_order(order)
        elif order_book.get_best_bid() >= order.price:
            self._execute_market_order(order)

    def _check_stop_order(self, order: Order):
        """Check if a stop order should be executed based on price."""
        symbol = order.symbol
        order_book = self.order_books.get(symbol)
        if not order_book:
            return
        if order.side == OrderSide.BUY:
            if order_book.get_best_ask() >= order.stop_price:
                order.order_type = OrderType.MARKET
                self._execute_market_order(order)
        elif order_book.get_best_bid() <= order.stop_price:
            order.order_type = OrderType.MARKET
            self._execute_market_order(order)

    def _check_stop_limit_order(self, order: Order):
        """Check if a stop-limit order should be activated or executed."""
        symbol = order.symbol
        order_book = self.order_books.get(symbol)
        if not order_book:
            return
        if order.side == OrderSide.BUY:
            if order_book.get_best_ask() >= order.stop_price:
                order.order_type = OrderType.LIMIT
                self._check_limit_order(order)
        elif order_book.get_best_bid() <= order.stop_price:
            order.order_type = OrderType.LIMIT
            self._check_limit_order(order)

    def _update_position(self, order: Order):
        """Update account position based on the executed order."""
        symbol = order.symbol
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0.0, 'avg_price': 0.0,
                'unrealized_pnl': 0.0}
        position = self.positions[symbol]
        if order.side == OrderSide.BUY:
            new_quantity = position['quantity'] + order.filled_quantity
            if new_quantity > 0:
                position['avg_price'] = (position['quantity'] * position[
                    'avg_price'] + order.filled_quantity * order.filled_price
                    ) / new_quantity
            position['quantity'] = new_quantity
        else:
            new_quantity = position['quantity'] - order.filled_quantity
            if position['quantity'] > 0 and new_quantity < 0:
                position['avg_price'] = order.filled_price
            elif position['quantity'] < 0 and new_quantity > 0:
                position['avg_price'] = order.filled_price
            position['quantity'] = new_quantity
        self._update_position_pnl(symbol)

    def _update_position_pnl(self, symbol: str):
        """Update the unrealized P&L for a position."""
        if symbol not in self.positions:
            return
        position = self.positions[symbol]
        quantity = position['quantity']
        if quantity == 0:
            position['unrealized_pnl'] = 0.0
            return
        order_book = self.order_books.get(symbol)
        if not order_book:
            return
        mid_price = order_book.mid_price
        if quantity > 0:
            position['unrealized_pnl'] = quantity * (mid_price - position[
                'avg_price'])
        else:
            position['unrealized_pnl'] = quantity * (position['avg_price'] -
                mid_price)

    def _record_transaction(self, order: Order):
        """Record a transaction in the history."""
        transaction = {'order_id': order.order_id, 'client_order_id': order
            .client_order_id, 'symbol': order.symbol, 'side': order.side.
            value, 'order_type': order.order_type.value, 'quantity': order.
            quantity, 'filled_quantity': order.filled_quantity, 'price':
            order.filled_price, 'status': order.status.value, 'timestamp': 
            order.filled_at.isoformat() if order.filled_at else None,
            'commission': self._calculate_commission(order)}
        self.transactions.append(transaction)

    def _calculate_commission(self, order: Order) ->float:
        """Calculate commission for an order."""
        lots = order.filled_quantity / 100000
        return lots * self.commission_per_lot

    @with_broker_api_resilience('get_account_summary')
    def get_account_summary(self) ->Dict[str, Any]:
        """Get a summary of the account status."""
        for symbol in self.positions:
            self._update_position_pnl(symbol)
        unrealized_pnl = sum(p['unrealized_pnl'] for p in self.positions.
            values())
        equity = self.balance + unrealized_pnl
        used_margin = 0.0
        for symbol, position in self.positions.items():
            if position['quantity'] != 0:
                price = self.prices.get(symbol, 0)
                margin_requirement = abs(position['quantity'] * price /
                    self.leverage)
                used_margin += margin_requirement
        return {'balance': self.balance, 'equity': equity, 'unrealized_pnl':
            unrealized_pnl, 'used_margin': used_margin, 'free_margin': 
            equity - used_margin, 'margin_level': equity / used_margin * 
            100 if used_margin > 0 else None, 'num_positions': len([p for p in
            self.positions.values() if p['quantity'] != 0]),
            'num_pending_orders': len([o for o in self.orders.values() if o
            .status == OrderStatus.PENDING])}

    @with_market_data_resilience('get_current_prices')
    def get_current_prices(self) ->Dict[str, Dict[str, float]]:
        """Get the current bid/ask prices for all symbols."""
        result = {}
        for symbol, order_book in self.order_books.items():
            result[symbol] = {'bid': order_book.get_best_bid(), 'ask':
                order_book.get_best_ask(), 'mid': order_book.mid_price,
                'spread': order_book.get_current_spread()}
        return result

    def place_order(self, order: Order) ->str:
        """
        Place a new order.
        
        Args:
            order: The order to place
            
        Returns:
            The order ID
        """
        if order.order_type == OrderType.MARKET:
            self._execute_market_order(order)
        else:
            pass
        self.orders[order.order_id] = order
        return order.order_id

    def cancel_order(self, order_id: str) ->bool:
        """
        Cancel a pending order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if successful, False otherwise
        """
        if order_id not in self.orders:
            return False
        order = self.orders[order_id]
        if order.status != OrderStatus.PENDING:
            return False
        order.status = OrderStatus.CANCELED
        return True

    def modify_order(self, order_id: str, new_price: Optional[float]=None,
        new_stop_price: Optional[float]=None, new_limit_price: Optional[
        float]=None) ->bool:
        """
        Modify a pending order.
        
        Args:
            order_id: ID of the order to modify
            new_price: New price for limit orders
            new_stop_price: New stop price for stop orders
            new_limit_price: New limit price for stop-limit orders
            
        Returns:
            True if successful, False otherwise
        """
        if order_id not in self.orders:
            return False
        order = self.orders[order_id]
        if order.status != OrderStatus.PENDING:
            return False
        if new_price is not None:
            order.price = new_price
        if new_stop_price is not None:
            order.stop_price = new_stop_price
        if new_limit_price is not None:
            order.limit_price = new_limit_price
        return True

    def close_position(self, symbol: str) ->Optional[str]:
        """
        Close an open position.
        
        Args:
            symbol: Symbol to close position for
            
        Returns:
            The order ID of the closing order, or None if no position
        """
        if symbol not in self.positions or self.positions[symbol]['quantity'
            ] == 0:
            return None
        position = self.positions[symbol]
        quantity = position['quantity']
        order = Order(symbol=symbol, side=OrderSide.SELL if quantity > 0 else
            OrderSide.BUY, order_type=OrderType.MARKET, quantity=abs(quantity))
        return self.place_order(order)

    def set_market_condition(self, condition: MarketConditionConfig) ->None:
        """
        Set a new market condition.
        
        Args:
            condition: The new market condition configuration
        """
        self.market_condition = condition
        for symbol in self.prices:
            base_spread = self.base_spreads[symbol]
            self.current_spreads[symbol
                ] = base_spread * condition.spread_factor
        for symbol, order_book in self.order_books.items():
            order_book.liquidity_profile = condition.liquidity_level
            order_book.update_spread(self.current_spreads[symbol])
        logger.info(f'Market condition updated to {condition.regime.value}')

    def trigger_custom_event(self, event: MarketEvent) ->None:
        """
        Trigger a custom market event.
        
        Args:
            event: The market event to trigger
        """
        self.active_events.append(event)
        logger.info(f'Custom event triggered: {event.description}')

    @with_broker_api_resilience('get_order_book_snapshot')
    def get_order_book_snapshot(self, symbol: str) ->Dict[str, List[Dict[
        str, float]]]:
        """
        Get a snapshot of the current order book.
        
        Args:
            symbol: The symbol to get the order book for
            
        Returns:
            Dictionary with bid and ask levels
        """
        if symbol not in self.order_books:
            return {'bids': [], 'asks': []}
        order_book = self.order_books[symbol]
        bids = [{'price': level.price, 'volume': level.volume} for level in
            order_book.bid_levels]
        asks = [{'price': level.price, 'volume': level.volume} for level in
            order_book.ask_levels]
        return {'bids': bids, 'asks': asks}

    @with_broker_api_resilience('get_trade_history')
    def get_trade_history(self) ->List[Dict[str, Any]]:
        """Get the trade history."""
        return self.transactions

    @with_broker_api_resilience('get_positions')
    def get_positions(self) ->Dict[str, Dict[str, float]]:
        """Get all open positions."""
        return {symbol: pos for symbol, pos in self.positions.items() if 
            pos['quantity'] != 0}

    @with_broker_api_resilience('get_pending_orders')
    def get_pending_orders(self) ->Dict[str, Order]:
        """Get all pending orders."""
        return {oid: order for oid, order in self.orders.items() if order.
            status == OrderStatus.PENDING}

    @with_broker_api_resilience('get_active_events')
    def get_active_events(self) ->List[Dict[str, Any]]:
        """Get all active market events."""
        return [{'type': event.event_type.value, 'description': event.
            description, 'impact': event.impact_magnitude, 'start_time':
            event.start_time.isoformat(), 'end_time': event.end_time.
            isoformat(), 'affected_symbols': event.affected_symbols} for
            event in self.active_events]
