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


class MarketRegimeType(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    CRISIS = "crisis"
    NORMAL = "normal"


class LiquidityLevel(Enum):
    VERY_LOW = "very_low"  # Extreme events, flash crashes
    LOW = "low"            # Off-hours, thin markets
    MEDIUM = "medium"      # Normal trading conditions
    HIGH = "high"          # Active market hours
    VERY_HIGH = "very_high"  # High activity periods, news releases


class MarketEventType(Enum):
    ECONOMIC_RELEASE = "economic_release"
    CENTRAL_BANK_DECISION = "central_bank_decision"
    GEOPOLITICAL_EVENT = "geopolitical_event"
    FLASH_CRASH = "flash_crash"
    LIQUIDITY_GAP = "liquidity_gap"
    TECHNICAL_BREAKOUT = "technical_breakout"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class Order:
    """
    Represents a forex order with all relevant details.
    """
    
    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        limit_price: Optional[float] = None,
        time_in_force: str = "GTC",  # Good Till Canceled
        order_id: Optional[str] = None,
        status: OrderStatus = OrderStatus.PENDING,
        created_at: Optional[datetime] = None,
        filled_at: Optional[datetime] = None,
        filled_price: Optional[float] = None,
        filled_quantity: float = 0.0,
        client_order_id: Optional[str] = None
    ):
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.limit_price = limit_price
        self.time_in_force = time_in_force
        self.order_id = order_id or f"ord_{int(datetime.now().timestamp()*1000)}"
        self.status = status
        self.created_at = created_at or datetime.now()
        self.filled_at = filled_at
        self.filled_price = filled_price
        self.filled_quantity = filled_quantity
        self.client_order_id = client_order_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert the order to a dictionary representation."""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "limit_price": self.limit_price,
            "time_in_force": self.time_in_force,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "filled_price": self.filled_price,
            "filled_quantity": self.filled_quantity,
        }


class MarketConditionConfig:
    """
    Configuration for market conditions in the simulation.
    """
    def __init__(
        self,
        regime: MarketRegimeType = MarketRegimeType.NORMAL,
        volatility_factor: float = 1.0,
        liquidity_level: LiquidityLevel = LiquidityLevel.MEDIUM,
        spread_factor: float = 1.0,
        event_frequency: float = 0.05,
        max_slippage: float = 0.0003,
        gap_probability: float = 0.01,
        session_volume_profile: Dict[str, float] = None
    ):
        self.regime = regime
        self.volatility_factor = volatility_factor
        self.liquidity_level = liquidity_level
        self.spread_factor = spread_factor
        self.event_frequency = event_frequency
        self.max_slippage = max_slippage
        self.gap_probability = gap_probability
        
        # Default session volume profile (% of daily volume)
        self.session_volume_profile = session_volume_profile or {
            "sydney": 0.1,
            "tokyo": 0.2,
            "london": 0.4,
            "new_york": 0.3
        }
        
    @classmethod
    def create_for_regime(cls, regime: MarketRegimeType) -> 'MarketConditionConfig':
        """
        Create a preset configuration for a specific market regime.
        
        Args:
            regime: The market regime type
            
        Returns:
            A MarketConditionConfig instance with settings appropriate for the regime
        """
        if regime == MarketRegimeType.TRENDING:
            return cls(
                regime=regime,
                volatility_factor=0.8,
                liquidity_level=LiquidityLevel.HIGH,
                spread_factor=0.9,
                event_frequency=0.03,
                gap_probability=0.005
            )
        elif regime == MarketRegimeType.RANGING:
            return cls(
                regime=regime,
                volatility_factor=0.7,
                liquidity_level=LiquidityLevel.MEDIUM,
                spread_factor=1.0,
                event_frequency=0.02,
                gap_probability=0.002
            )
        elif regime == MarketRegimeType.VOLATILE:
            return cls(
                regime=regime,
                volatility_factor=2.0,
                liquidity_level=LiquidityLevel.MEDIUM,
                spread_factor=1.5,
                event_frequency=0.1,
                gap_probability=0.03
            )
        elif regime == MarketRegimeType.BREAKOUT:
            return cls(
                regime=regime,
                volatility_factor=1.5,
                liquidity_level=LiquidityLevel.LOW,
                spread_factor=1.3,
                event_frequency=0.05,
                gap_probability=0.04
            )
        elif regime == MarketRegimeType.CRISIS:
            return cls(
                regime=regime,
                volatility_factor=3.0,
                liquidity_level=LiquidityLevel.VERY_LOW,
                spread_factor=3.0,
                event_frequency=0.2,
                gap_probability=0.08,
                session_volume_profile={
                    "sydney": 0.05,
                    "tokyo": 0.15,
                    "london": 0.5,
                    "new_york": 0.3
                }
            )
        else:  # NORMAL
            return cls(
                regime=MarketRegimeType.NORMAL,
                volatility_factor=1.0,
                liquidity_level=LiquidityLevel.MEDIUM,
                spread_factor=1.0,
                event_frequency=0.05,
                gap_probability=0.01
            )


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
    def __init__(
        self, 
        mid_price: float,
        spread_pips: float,
        num_levels: int = 10,
        liquidity_profile: LiquidityLevel = LiquidityLevel.MEDIUM
    ):
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
        # Define pip size based on typical FX convention
        pip_size = 0.0001  # Standard for most pairs, would be 0.01 for JPY pairs
        
        # Calculate bid/ask prices
        spread_in_price = self.spread_pips * pip_size
        bid_price = self.mid_price - (spread_in_price / 2)
        ask_price = self.mid_price + (spread_in_price / 2)
        
        # Set volume profile based on liquidity
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
        else:  # VERY_HIGH
            base_volume = 10.0
            volume_decay = 1.2
            
        # Generate bid levels (descending price)
        self.bid_levels = []
        for i in range(self.num_levels):
            level_price = bid_price - (i * pip_size)
            # Volume increases closer to the mid price with some randomness
            level_volume = base_volume * (volume_decay ** i) * (0.8 + 0.4 * random.random())
            self.bid_levels.append(OrderBookLevel(level_price, level_volume))
            
        # Generate ask levels (ascending price)
        self.ask_levels = []
        for i in range(self.num_levels):
            level_price = ask_price + (i * pip_size)
            # Volume increases closer to the mid price with some randomness
            level_volume = base_volume * (volume_decay ** i) * (0.8 + 0.4 * random.random())
            self.ask_levels.append(OrderBookLevel(level_price, level_volume))
    
    def update_mid_price(self, new_mid_price: float):
        """
        Update the order book with a new mid price.
        
        Args:
            new_mid_price: The new mid price
        """
        self.mid_price = new_mid_price
        self._generate_order_book()
        
    def update_spread(self, new_spread_pips: float):
        """
        Update the order book with a new spread.
        
        Args:
            new_spread_pips: The new spread in pips
        """
        self.spread_pips = new_spread_pips
        self._generate_order_book()
    
    def get_best_bid(self) -> float:
        """Get the current best bid price."""
        return self.bid_levels[0].price if self.bid_levels else 0
        
    def get_best_ask(self) -> float:
        """Get the current best ask price."""
        return self.ask_levels[0].price if self.ask_levels else 0
        
    def get_current_spread(self) -> float:
        """Get the current spread in price terms."""
        return self.get_best_ask() - self.get_best_bid()


class MarketEvent:
    """
    Represents a market event that affects price, volatility, and liquidity.
    """
    def __init__(
        self,
        event_type: MarketEventType,
        impact_magnitude: float,  # 0.0 to 1.0 where 1.0 is extreme
        duration_minutes: int,
        description: str,
        affected_symbols: List[str]
    ):
        self.event_type = event_type
        self.impact_magnitude = impact_magnitude
        self.duration_minutes = duration_minutes
        self.description = description
        self.affected_symbols = affected_symbols
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(minutes=duration_minutes)
        
    def is_active(self, current_time: datetime) -> bool:
        """Check if the event is currently active."""
        return self.start_time <= current_time <= self.end_time
        
    def get_price_impact_factor(self) -> float:
        """Get the impact factor on price."""
        if self.event_type == MarketEventType.FLASH_CRASH:
            return -0.1 * self.impact_magnitude
        elif self.event_type == MarketEventType.LIQUIDITY_GAP:
            # Can be positive or negative
            return 0.05 * self.impact_magnitude * (1 if random.random() > 0.5 else -1)
        elif self.event_type == MarketEventType.TECHNICAL_BREAKOUT:
            # Can be positive or negative
            return 0.03 * self.impact_magnitude * (1 if random.random() > 0.5 else -1)
        elif self.event_type == MarketEventType.CENTRAL_BANK_DECISION:
            return 0.02 * self.impact_magnitude * (1 if random.random() > 0.5 else -1)
        elif self.event_type == MarketEventType.ECONOMIC_RELEASE:
            return 0.01 * self.impact_magnitude * (1 if random.random() > 0.5 else -1)
        else:
            return 0.005 * self.impact_magnitude * (1 if random.random() > 0.5 else -1)
            
    def get_volatility_impact_factor(self) -> float:
        """Get the impact factor on volatility."""
        if self.event_type == MarketEventType.FLASH_CRASH:
            return 3.0 * self.impact_magnitude
        elif self.event_type == MarketEventType.CENTRAL_BANK_DECISION:
            return 2.0 * self.impact_magnitude
        elif self.event_type == MarketEventType.ECONOMIC_RELEASE:
            return 1.5 * self.impact_magnitude
        else:
            return 1.0 * self.impact_magnitude
            
    def get_spread_impact_factor(self) -> float:
        """Get the impact factor on spread."""
        if self.event_type == MarketEventType.FLASH_CRASH:
            return 5.0 * self.impact_magnitude
        elif self.event_type == MarketEventType.LIQUIDITY_GAP:
            return 3.0 * self.impact_magnitude
        elif self.event_type == MarketEventType.CENTRAL_BANK_DECISION:
            return 2.0 * self.impact_magnitude
        else:
            return 1.0 * self.impact_magnitude
    
    def get_liquidity_impact(self) -> LiquidityLevel:
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
    
    def __init__(
        self,
        initial_prices: Dict[str, float] = None,
        base_spreads: Dict[str, float] = None,
        initial_condition: MarketConditionConfig = None,
        balance: float = 10000.0,
        leverage: float = 100.0,
        data_source: Optional[pd.DataFrame] = None,
        replay_mode: bool = False,
        commission_per_lot: float = 7.0,
        time_acceleration: float = 1.0  # 1.0 = real-time, >1.0 = faster
    ):
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
        self.initial_prices = initial_prices or {
            "EUR/USD": 1.1000,
            "GBP/USD": 1.3000,
            "USD/JPY": 110.00,
            "AUD/USD": 0.7000,
            "USD/CAD": 1.3000
        }
        self.prices = self.initial_prices.copy()
        
        self.base_spreads = base_spreads or {
            "EUR/USD": 1.0,  # 1 pip
            "GBP/USD": 1.5,  # 1.5 pips
            "USD/JPY": 1.0,  # 1 pip
            "AUD/USD": 1.2,  # 1.2 pips
            "USD/CAD": 1.5   # 1.5 pips
        }
        
        self.current_spreads = {s: spread for s, spread in self.base_spreads.items()}
        self.market_condition = initial_condition or MarketConditionConfig()
        self.balance = balance
        self.initial_balance = balance
        self.leverage = leverage
        self.data_source = data_source
        self.replay_mode = replay_mode
        self.commission_per_lot = commission_per_lot
        self.time_acceleration = time_acceleration
        
        # Current time in the simulation
        self.current_time = datetime.now()
        self.start_time = self.current_time
        self.simulation_time_elapsed = timedelta(0)
        
        # Active orders and positions
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Dict[str, float]] = {}
        
        # Order book for each symbol
        self.order_books: Dict[str, OrderBook] = {}
        
        # Active market events
        self.active_events: List[MarketEvent] = []
        
        # Transaction history
        self.transactions: List[Dict[str, Any]] = []
        
        # Initialize order books
        self._initialize_order_books()
        
        logger.info(f"ForexBrokerSimulator initialized with {len(self.prices)} symbols")
        
    def _initialize_order_books(self):
        """Initialize order books for all symbols."""
        for symbol, price in self.prices.items():
            spread = self.current_spreads[symbol]
            liquidity = self.market_condition.liquidity_level
            self.order_books[symbol] = OrderBook(price, spread, liquidity_profile=liquidity)
    
    def advance_time(self, seconds: float) -> None:
        """
        Advance the simulation time by the specified number of seconds.
        
        Args:
            seconds: Number of seconds to advance (will be multiplied by time_acceleration)
        """
        actual_seconds = seconds * self.time_acceleration
        self.current_time += timedelta(seconds=actual_seconds)
        self.simulation_time_elapsed += timedelta(seconds=actual_seconds)
        
        # Update prices based on time passed
        if not self.replay_mode:
            self._update_prices(actual_seconds)
        
        # Process pending orders
        self._process_pending_orders()
        
        # Check for expired events
        self._update_events()
        
        # Generate new events
        if random.random() < self.market_condition.event_frequency * (actual_seconds / 3600):
            self._generate_market_event()
    
    def _update_prices(self, seconds_elapsed: float):
        """
        Update prices based on time elapsed and current market conditions.
        
        Args:
            seconds_elapsed: Time elapsed in seconds
        """
        # Base volatility per hour as percentage
        base_hourly_volatility = {
            "EUR/USD": 0.0010,  # 0.10%
            "GBP/USD": 0.0015,  # 0.15%
            "USD/JPY": 0.0012,  # 0.12%
            "AUD/USD": 0.0018,  # 0.18%
            "USD/CAD": 0.0014   # 0.14%
        }
        
        # Scale by regime volatility and convert to per-second
        for symbol, price in self.prices.items():
            hourly_vol = base_hourly_volatility.get(symbol, 0.0015)
            vol_scaling = self.market_condition.volatility_factor
            
            # Add extra volatility from active events
            for event in self.active_events:
                if symbol in event.affected_symbols:
                    vol_scaling *= event.get_volatility_impact_factor()
            
            # Convert to per-second volatility
            second_vol = hourly_vol * vol_scaling / np.sqrt(3600)
            
            # Calculate price change using a random walk model
            drift = 0
            diffusion = np.random.normal(0, second_vol) * price
            
            # Add event price impacts
            event_impact = 0
            for event in self.active_events:
                if symbol in event.affected_symbols:
                    event_impact += price * event.get_price_impact_factor()
            
            # Update price
            new_price = price + drift + diffusion + event_impact
            self.prices[symbol] = max(0.0001, new_price)  # Ensure price is positive
            
            # Update spread based on market conditions
            base_spread = self.base_spreads[symbol]
            spread_factor = self.market_condition.spread_factor
            
            # Adjust spread for active events
            for event in self.active_events:
                if symbol in event.affected_symbols:
                    spread_factor *= event.get_spread_impact_factor()
            
            # Update current spread
            self.current_spreads[symbol] = base_spread * spread_factor
            
            # Update order book
            self.order_books[symbol].update_mid_price(self.prices[symbol])
            self.order_books[symbol].update_spread(self.current_spreads[symbol])
            
            # Check for price gaps
            gap_probability = self.market_condition.gap_probability * (seconds_elapsed / 3600)
            if random.random() < gap_probability:
                gap_size = random.uniform(0.0005, 0.003) * price  # 0.05% to 0.3%
                direction = 1 if random.random() > 0.5 else -1
                gap_price = price + (gap_size * direction)
                self.prices[symbol] = max(0.0001, gap_price)
                self.order_books[symbol].update_mid_price(self.prices[symbol])
    
    def _update_events(self):
        """Update and expire market events."""
        active_events = []
        for event in self.active_events:
            if event.is_active(self.current_time):
                active_events.append(event)
            else:
                logger.info(f"Market event expired: {event.description}")
                
        self.active_events = active_events
    
    def _generate_market_event(self):
        """Generate a random market event."""
        event_types = list(MarketEventType)
        weights = [0.4, 0.2, 0.1, 0.05, 0.1, 0.15]  # Probability distribution
        
        event_type = random.choices(event_types, weights=weights, k=1)[0]
        
        # Determine impact based on event type
        if event_type == MarketEventType.FLASH_CRASH:
            impact = random.uniform(0.7, 1.0)
            duration = random.randint(5, 30)  # 5-30 minutes
        elif event_type == MarketEventType.LIQUIDITY_GAP:
            impact = random.uniform(0.5, 0.9)
            duration = random.randint(5, 60)  # 5-60 minutes
        elif event_type == MarketEventType.CENTRAL_BANK_DECISION:
            impact = random.uniform(0.4, 0.8)
            duration = random.randint(60, 240)  # 1-4 hours
        elif event_type == MarketEventType.ECONOMIC_RELEASE:
            impact = random.uniform(0.2, 0.7)
            duration = random.randint(15, 90)  # 15-90 minutes
        else:
            impact = random.uniform(0.1, 0.5)
            duration = random.randint(30, 180)  # 30-180 minutes
            
        # Select affected symbols
        all_symbols = list(self.prices.keys())
        num_affected = random.randint(1, max(1, len(all_symbols) // 2))
        affected_symbols = random.sample(all_symbols, num_affected)
        
        # Create description
        descriptions = {
            MarketEventType.FLASH_CRASH: "Sudden market flash crash",
            MarketEventType.LIQUIDITY_GAP: "Significant liquidity gap",
            MarketEventType.CENTRAL_BANK_DECISION: "Central bank rate decision",
            MarketEventType.ECONOMIC_RELEASE: "Major economic data release",
            MarketEventType.GEOPOLITICAL_EVENT: "Breaking geopolitical development",
            MarketEventType.TECHNICAL_BREAKOUT: "Technical level breakout"
        }
        description = descriptions[event_type] + f" affecting {', '.join(affected_symbols)}"
        
        # Create and add the event
        event = MarketEvent(
            event_type=event_type,
            impact_magnitude=impact,
            duration_minutes=duration,
            description=description,
            affected_symbols=affected_symbols
        )
        
        self.active_events.append(event)
        logger.info(f"Generated market event: {description}, impact: {impact:.2f}, duration: {duration} minutes")
        
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
    
    def _execute_market_order(self, order: Order) -> bool:
        """
        Execute a market order with realistic slippage and partial fills.
        
        Args:
            order: The order to execute
            
        Returns:
            True if execution was successful
        """
        symbol = order.symbol
        
        # Get current order book
        order_book = self.order_books.get(symbol)
        if not order_book:
            logger.error(f"No order book for symbol {symbol}")
            order.status = OrderStatus.REJECTED
            return False
            
        # Determine execution price with slippage
        if order.side == OrderSide.BUY:
            base_price = order_book.get_best_ask()
            # Calculate slippage based on order size and liquidity
            slippage = self._calculate_slippage(order, is_buy=True)
            execution_price = base_price * (1 + slippage)
        else:  # SELL
            base_price = order_book.get_best_bid()
            # Calculate slippage based on order size and liquidity
            slippage = self._calculate_slippage(order, is_buy=False)
            execution_price = base_price * (1 - slippage)
        
        # Check if we should do a partial fill
        fill_quantity = order.quantity
        is_partial = False
        
        # Large orders may get partial fills in low liquidity
        if (self.market_condition.liquidity_level in [LiquidityLevel.VERY_LOW, LiquidityLevel.LOW] and 
                order.quantity > 5.0):  # Threshold for "large" order
            fill_ratio = random.uniform(0.6, 0.95)
            fill_quantity = order.quantity * fill_ratio
            is_partial = True
            
        # Update order
        if is_partial:
            order.status = OrderStatus.PARTIALLY_FILLED
            order.filled_quantity = fill_quantity
        else:
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            
        order.filled_price = execution_price
        order.filled_at = self.current_time
        
        # Update position
        self._update_position(order)
        
        # Record transaction
        self._record_transaction(order)
        
        return True
    
    def _calculate_slippage(self, order: Order, is_buy: bool) -> float:
        """
        Calculate realistic slippage based on order size, market conditions, and events.
        
        Args:
            order: The order
            is_buy: Whether this is a buy order
            
        Returns:
            Slippage as a decimal percentage
        """
        symbol = order.symbol
        
        # Base slippage according to liquidity level
        base_slippage = {
            LiquidityLevel.VERY_LOW: 0.0005,  # 0.05%
            LiquidityLevel.LOW: 0.0003,       # 0.03%
            LiquidityLevel.MEDIUM: 0.0001,    # 0.01%
            LiquidityLevel.HIGH: 0.00005,     # 0.005%
            LiquidityLevel.VERY_HIGH: 0.00002 # 0.002%
        }
        
        slippage = base_slippage.get(self.market_condition.liquidity_level, 0.0001)
        
        # Scale by order size (larger orders = more slippage)
        # Assuming standard lot = 100,000 units
        size_factor = min(5.0, 1.0 + (order.quantity / 10.0))
        slippage *= size_factor
        
        # Add extra slippage from active events
        for event in self.active_events:
            if symbol in event.affected_symbols:
                slippage *= (1.0 + event.get_volatility_impact_factor() / 2)
        
        # Cap at maximum slippage
        return min(slippage, self.market_condition.max_slippage)
    
    def _check_limit_order(self, order: Order):
        """Check if a limit order should be executed based on price."""
        symbol = order.symbol
        order_book = self.order_books.get(symbol)
        
        if not order_book:
            return
            
        if order.side == OrderSide.BUY:
            # Buy limit executes when ask price <= limit price
            if order_book.get_best_ask() <= order.price:
                self._execute_market_order(order)
        else:  # SELL
            # Sell limit executes when bid price >= limit price
            if order_book.get_best_bid() >= order.price:
                self._execute_market_order(order)
    
    def _check_stop_order(self, order: Order):
        """Check if a stop order should be executed based on price."""
        symbol = order.symbol
        order_book = self.order_books.get(symbol)
        
        if not order_book:
            return
            
        if order.side == OrderSide.BUY:
            # Buy stop executes when ask price >= stop price
            if order_book.get_best_ask() >= order.stop_price:
                # Convert to market order
                order.order_type = OrderType.MARKET
                self._execute_market_order(order)
        else:  # SELL
            # Sell stop executes when bid price <= stop price
            if order_book.get_best_bid() <= order.stop_price:
                # Convert to market order
                order.order_type = OrderType.MARKET
                self._execute_market_order(order)
    
    def _check_stop_limit_order(self, order: Order):
        """Check if a stop-limit order should be activated or executed."""
        symbol = order.symbol
        order_book = self.order_books.get(symbol)
        
        if not order_book:
            return
            
        # Check if stop price has been reached
        if order.side == OrderSide.BUY:
            # Buy stop-limit activates when ask price >= stop price
            if order_book.get_best_ask() >= order.stop_price:
                # Convert to limit order
                order.order_type = OrderType.LIMIT
                self._check_limit_order(order)
        else:  # SELL
            # Sell stop-limit activates when bid price <= stop price
            if order_book.get_best_bid() <= order.stop_price:
                # Convert to limit order
                order.order_type = OrderType.LIMIT
                self._check_limit_order(order)
    
    def _update_position(self, order: Order):
        """Update account position based on the executed order."""
        symbol = order.symbol
        
        # Initialize position if not exists
        if symbol not in self.positions:
            self.positions[symbol] = {
                "quantity": 0.0,
                "avg_price": 0.0,
                "unrealized_pnl": 0.0
            }
            
        position = self.positions[symbol]
        
        # Calculate position change
        if order.side == OrderSide.BUY:
            new_quantity = position["quantity"] + order.filled_quantity
            
            # Update average price (weighted average)
            if new_quantity > 0:
                position["avg_price"] = (position["quantity"] * position["avg_price"] + 
                                       order.filled_quantity * order.filled_price) / new_quantity
            
            position["quantity"] = new_quantity
            
        else:  # SELL
            # Reduce position and calculate realized P&L
            new_quantity = position["quantity"] - order.filled_quantity
            
            # If position direction changes, reset average price
            if position["quantity"] > 0 and new_quantity < 0:
                position["avg_price"] = order.filled_price
            elif position["quantity"] < 0 and new_quantity > 0:
                position["avg_price"] = order.filled_price
                
            position["quantity"] = new_quantity
        
        # Update unrealized PnL
        self._update_position_pnl(symbol)
    
    def _update_position_pnl(self, symbol: str):
        """Update the unrealized P&L for a position."""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        quantity = position["quantity"]
        
        if quantity == 0:
            position["unrealized_pnl"] = 0.0
            return
        
        # Get current mid price
        order_book = self.order_books.get(symbol)
        if not order_book:
            return
            
        mid_price = order_book.mid_price
        
        # Calculate P&L
        if quantity > 0:  # Long position
            position["unrealized_pnl"] = quantity * (mid_price - position["avg_price"])
        else:  # Short position
            position["unrealized_pnl"] = quantity * (position["avg_price"] - mid_price)
    
    def _record_transaction(self, order: Order):
        """Record a transaction in the history."""
        transaction = {
            "order_id": order.order_id,
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "quantity": order.quantity,
            "filled_quantity": order.filled_quantity,
            "price": order.filled_price,
            "status": order.status.value,
            "timestamp": order.filled_at.isoformat() if order.filled_at else None,
            "commission": self._calculate_commission(order)
        }
        
        self.transactions.append(transaction)
    
    def _calculate_commission(self, order: Order) -> float:
        """Calculate commission for an order."""
        # Standard lot is 100,000 units
        lots = order.filled_quantity / 100000
        return lots * self.commission_per_lot
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get a summary of the account status."""
        # Update unrealized P&L for all positions
        for symbol in self.positions:
            self._update_position_pnl(symbol)
            
        # Calculate total equity
        unrealized_pnl = sum(p["unrealized_pnl"] for p in self.positions.values())
        equity = self.balance + unrealized_pnl
        
        # Calculate used margin
        used_margin = 0.0
        for symbol, position in self.positions.items():
            if position["quantity"] != 0:
                price = self.prices.get(symbol, 0)
                margin_requirement = abs(position["quantity"] * price / self.leverage)
                used_margin += margin_requirement
                
        return {
            "balance": self.balance,
            "equity": equity,
            "unrealized_pnl": unrealized_pnl,
            "used_margin": used_margin,
            "free_margin": equity - used_margin,
            "margin_level": (equity / used_margin * 100) if used_margin > 0 else None,
            "num_positions": len([p for p in self.positions.values() if p["quantity"] != 0]),
            "num_pending_orders": len([o for o in self.orders.values() if o.status == OrderStatus.PENDING])
        }
        
    def get_current_prices(self) -> Dict[str, Dict[str, float]]:
        """Get the current bid/ask prices for all symbols."""
        result = {}
        for symbol, order_book in self.order_books.items():
            result[symbol] = {
                "bid": order_book.get_best_bid(),
                "ask": order_book.get_best_ask(),
                "mid": order_book.mid_price,
                "spread": order_book.get_current_spread()
            }
        return result
    
    def place_order(self, order: Order) -> str:
        """
        Place a new order.
        
        Args:
            order: The order to place
            
        Returns:
            The order ID
        """
        # Process market orders immediately, queue other types
        if order.order_type == OrderType.MARKET:
            self._execute_market_order(order)
        else:
            # Just queue the order for processing
            pass
            
        # Store the order
        self.orders[order.order_id] = order
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
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
    
    def modify_order(self, order_id: str, new_price: Optional[float] = None,
                    new_stop_price: Optional[float] = None, 
                    new_limit_price: Optional[float] = None) -> bool:
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
            
        # Update order parameters
        if new_price is not None:
            order.price = new_price
        if new_stop_price is not None:
            order.stop_price = new_stop_price
        if new_limit_price is not None:
            order.limit_price = new_limit_price
            
        return True
    
    def close_position(self, symbol: str) -> Optional[str]:
        """
        Close an open position.
        
        Args:
            symbol: Symbol to close position for
            
        Returns:
            The order ID of the closing order, or None if no position
        """
        if symbol not in self.positions or self.positions[symbol]["quantity"] == 0:
            return None
            
        position = self.positions[symbol]
        quantity = position["quantity"]
        
        # Create closing order
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL if quantity > 0 else OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=abs(quantity)
        )
        
        # Place the order
        return self.place_order(order)
    
    def set_market_condition(self, condition: MarketConditionConfig) -> None:
        """
        Set a new market condition.
        
        Args:
            condition: The new market condition configuration
        """
        self.market_condition = condition
        
        # Update spreads
        for symbol in self.prices:
            base_spread = self.base_spreads[symbol]
            self.current_spreads[symbol] = base_spread * condition.spread_factor
            
        # Update order books
        for symbol, order_book in self.order_books.items():
            order_book.liquidity_profile = condition.liquidity_level
            order_book.update_spread(self.current_spreads[symbol])
            
        logger.info(f"Market condition updated to {condition.regime.value}")
    
    def trigger_custom_event(self, event: MarketEvent) -> None:
        """
        Trigger a custom market event.
        
        Args:
            event: The market event to trigger
        """
        self.active_events.append(event)
        logger.info(f"Custom event triggered: {event.description}")
    
    def get_order_book_snapshot(self, symbol: str) -> Dict[str, List[Dict[str, float]]]:
        """
        Get a snapshot of the current order book.
        
        Args:
            symbol: The symbol to get the order book for
            
        Returns:
            Dictionary with bid and ask levels
        """
        if symbol not in self.order_books:
            return {"bids": [], "asks": []}
            
        order_book = self.order_books[symbol]
        
        bids = [{"price": level.price, "volume": level.volume} for level in order_book.bid_levels]
        asks = [{"price": level.price, "volume": level.volume} for level in order_book.ask_levels]
        
        return {"bids": bids, "asks": asks}
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get the trade history."""
        return self.transactions
    
    def get_positions(self) -> Dict[str, Dict[str, float]]:
        """Get all open positions."""
        return {symbol: pos for symbol, pos in self.positions.items() if pos["quantity"] != 0}
    
    def get_pending_orders(self) -> Dict[str, Order]:
        """Get all pending orders."""
        return {oid: order for oid, order in self.orders.items() 
                if order.status == OrderStatus.PENDING}
    
    def get_active_events(self) -> List[Dict[str, Any]]:
        """Get all active market events."""
        return [
            {
                "type": event.event_type.value,
                "description": event.description,
                "impact": event.impact_magnitude,
                "start_time": event.start_time.isoformat(),
                "end_time": event.end_time.isoformat(),
                "affected_symbols": event.affected_symbols
            }
            for event in self.active_events
        ]
