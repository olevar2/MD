"""
Market Data Simulator for Paper Trading.

Simulates realistic forex market data including:
- Bid/Ask price generation
- Spread dynamics
- Volume profiles
- Market session effects
- Market volatility regimes
"""
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
logger = logging.getLogger(__name__)


from utils.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class MarketRegime(str, Enum):
    """Market volatility regime types."""
    LOW_VOLATILITY = 'low_volatility'
    NORMAL = 'normal'
    HIGH_VOLATILITY = 'high_volatility'
    EXTREME = 'extreme'


@dataclass
class MarketProfile:
    """Configuration for market behavior."""
    base_spread: float
    min_spread: float
    max_spread: float
    volatility_profile: Dict[MarketRegime, float]
    session_spreads: Dict[str, float]
    typical_volume: float
    tick_size: float
    price_decimals: int


class MarketDataSimulator:
    """
    Simulates realistic forex market data.
    
    Features:
    - Session-aware spread dynamics
    - Volume profile simulation
    - Realistic price movements
    - Market regime transitions
    """

    def __init__(self, symbols: List[str], historical_data: Optional[Dict[
        str, pd.DataFrame]]=None, session_config: Optional[Dict[str, Dict[
        str, str]]]=None):
    """
      init  .
    
    Args:
        symbols: Description of symbols
        historical_data: Description of historical_data
        pd.DataFrame]]: Description of pd.DataFrame]]
        session_config: Description of session_config
        Dict[
        str: Description of Dict[
        str
        str]]]: Description of str]]]
    
    """

        self.symbols = symbols
        self.historical_data = historical_data or {}
        self.session_config = session_config or {'sydney': {'open': '22:00',
            'close': '07:00'}, 'tokyo': {'open': '00:00', 'close': '09:00'},
            'london': {'open': '08:00', 'close': '17:00'}, 'new_york': {
            'open': '13:00', 'close': '22:00'}}
        self.market_profiles = self._initialize_market_profiles()
        self.current_prices: Dict[str, Dict[str, float]] = {}
        self.current_regime = MarketRegime.NORMAL
        self.regime_duration = 0
        self.last_update = datetime.utcnow()
        self.tick_count = 0

    def _initialize_market_profiles(self) ->Dict[str, MarketProfile]:
        """Initialize market profiles for each symbol."""
        profiles = {}
        profiles['EUR/USD'] = MarketProfile(base_spread=1.0, min_spread=0.8,
            max_spread=3.0, volatility_profile={MarketRegime.LOW_VOLATILITY:
            0.5, MarketRegime.NORMAL: 1.0, MarketRegime.HIGH_VOLATILITY: 
            2.0, MarketRegime.EXTREME: 4.0}, session_spreads={'sydney': 1.3,
            'tokyo': 1.2, 'london': 1.0, 'new_york': 1.0}, typical_volume=
            1000000, tick_size=1e-05, price_decimals=5)
        profiles['GBP/USD'] = MarketProfile(base_spread=1.5, min_spread=1.2,
            max_spread=4.0, volatility_profile={MarketRegime.LOW_VOLATILITY:
            0.6, MarketRegime.NORMAL: 1.0, MarketRegime.HIGH_VOLATILITY: 
            2.5, MarketRegime.EXTREME: 5.0}, session_spreads={'sydney': 1.5,
            'tokyo': 1.4, 'london': 1.0, 'new_york': 1.1}, typical_volume=
            800000, tick_size=1e-05, price_decimals=5)
        return profiles

    def generate_tick(self, timestamp: Optional[datetime]=None) ->Dict[str,
        Dict[str, Any]]:
        """
        Generate a new market data tick.
        
        Args:
            timestamp: Optional timestamp for the tick
            
        Returns:
            Dictionary containing market data for all symbols
        """
        timestamp = timestamp or datetime.utcnow()
        tick_data = {}
        self._update_market_regime(timestamp)
        for symbol in self.symbols:
            if symbol in self.market_profiles:
                tick_data[symbol] = self._generate_symbol_data(symbol,
                    timestamp)
        self.current_prices.update({s: {'bid': d['bid'], 'ask': d['ask']} for
            s, d in tick_data.items()})
        self.last_update = timestamp
        self.tick_count += 1
        return tick_data

    def _generate_symbol_data(self, symbol: str, timestamp: datetime) ->Dict[
        str, Any]:
        """Generate market data for a single symbol."""
        profile = self.market_profiles[symbol]
        session = self._get_current_session(timestamp)
        session_multiplier = profile.session_spreads.get(session, 1.0
            ) if session else 1.2
        base_spread = profile.base_spread * session_multiplier
        volatility_factor = profile.volatility_profile[self.current_regime]
        spread = min(profile.max_spread, max(profile.min_spread, 
            base_spread * volatility_factor))
        if symbol in self.current_prices:
            last_price = (self.current_prices[symbol]['bid'] + self.
                current_prices[symbol]['ask']) / 2
            volatility = self._get_symbol_volatility(symbol)
            price_change = np.random.normal(0, volatility * profile.tick_size)
            mid_price = last_price + price_change
        else:
            mid_price = self._get_reference_price(symbol)
        half_spread = spread * profile.tick_size / 2
        bid = round(mid_price - half_spread, profile.price_decimals)
        ask = round(mid_price + half_spread, profile.price_decimals)
        volume = self._generate_volume(symbol, session)
        return {'symbol': symbol, 'timestamp': timestamp, 'bid': bid, 'ask':
            ask, 'spread': spread, 'volume': volume, 'session': session,
            'regime': self.current_regime}

    def _update_market_regime(self, timestamp: datetime) ->None:
        """Update market regime state."""
        self.regime_duration += 1
        change_prob = min(0.001 * self.regime_duration, 0.05)
        if np.random.random() < change_prob:
            if self.current_regime == MarketRegime.NORMAL:
                self.current_regime = np.random.choice([MarketRegime.
                    LOW_VOLATILITY, MarketRegime.HIGH_VOLATILITY,
                    MarketRegime.EXTREME], p=[0.4, 0.4, 0.2])
            else:
                self.current_regime = np.random.choice([MarketRegime.NORMAL,
                    self.current_regime], p=[0.7, 0.3])
            self.regime_duration = 0
            logger.info(f'Market regime changed to {self.current_regime}')

    def _get_current_session(self, timestamp: datetime) ->Optional[str]:
        """Determine current active trading session."""
        time = timestamp.strftime('%H:%M')
        active_sessions = []
        for session, times in self.session_config.items():
            session_open = datetime.strptime(times['open'], '%H:%M').time()
            session_close = datetime.strptime(times['close'], '%H:%M').time()
            current_time = datetime.strptime(time, '%H:%M').time()
            if session_open > session_close:
                is_active = (current_time >= session_open or current_time <=
                    session_close)
            else:
                is_active = session_open <= current_time <= session_close
            if is_active:
                active_sessions.append(session)
        if 'london' in active_sessions:
            return 'london'
        elif 'new_york' in active_sessions:
            return 'new_york'
        elif 'tokyo' in active_sessions:
            return 'tokyo'
        elif active_sessions:
            return active_sessions[0]
        return None

    def _get_symbol_volatility(self, symbol: str) ->float:
        """Get current volatility for a symbol."""
        if symbol not in self.market_profiles:
            return 1.0
        profile = self.market_profiles[symbol]
        base_volatility = 1.0
        regime_factor = profile.volatility_profile[self.current_regime]
        return base_volatility * regime_factor

    def _generate_volume(self, symbol: str, session: Optional[str]) ->float:
        """Generate realistic trading volume."""
        if symbol not in self.market_profiles:
            return 0.0
        profile = self.market_profiles[symbol]
        base_volume = profile.typical_volume
        session_factors = {'sydney': 0.5, 'tokyo': 0.8, 'london': 1.0,
            'new_york': 1.0}
        session_factor = session_factors.get(session, 0.3)
        regime_factors = {MarketRegime.LOW_VOLATILITY: 0.7, MarketRegime.
            NORMAL: 1.0, MarketRegime.HIGH_VOLATILITY: 1.5, MarketRegime.
            EXTREME: 2.0}
        regime_factor = regime_factors[self.current_regime]
        volume = base_volume * session_factor * regime_factor
        volume *= np.random.lognormal(0, 0.2)
        return round(volume, -3)

    def _get_reference_price(self, symbol: str) ->float:
        """Get reference price for a symbol."""
        if symbol in self.historical_data:
            return self.historical_data[symbol].iloc[-1]['close']
        reference_prices = {'EUR/USD': 1.1, 'GBP/USD': 1.25, 'USD/JPY': 
            110.0, 'AUD/USD': 0.75, 'USD/CHF': 0.92, 'USD/CAD': 1.28,
            'NZD/USD': 0.7}
        return reference_prices.get(symbol, 1.0)

    @with_market_data_resilience('get_current_market_state')
    def get_current_market_state(self) ->Dict[str, Any]:
        """Get current market state information."""
        return {'timestamp': self.last_update, 'regime': self.
            current_regime, 'regime_duration': self.regime_duration,
            'active_session': self._get_current_session(datetime.utcnow()),
            'tick_count': self.tick_count, 'prices': self.current_prices}
