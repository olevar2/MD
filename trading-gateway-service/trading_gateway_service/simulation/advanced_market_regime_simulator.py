"""
Advanced Market Regime Simulator for Forex Trading.

This module provides enhanced market simulation capabilities for realistic
training of reinforcement learning agents. It generates synthetic market data
with configurable regimes, conditions, and events.
"""
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
import random
import logging
from scipy import stats
from trading_gateway_service.simulation.forex_broker_simulator import ForexBrokerSimulator, MarketRegimeType, Order
from trading_gateway_service.simulation.news_sentiment_simulator import NewsAndSentimentSimulator, NewsEvent, NewsImpactLevel, SentimentLevel
from core_foundations.utils.logger import get_logger
from core_foundations.models.financial_instruments import SymbolInfo
logger = get_logger(__name__)
from trading_gateway_service.error.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from trading_gateway_service.resilience.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class MarketCondition(str, Enum):
    """Detailed market conditions beyond basic regimes."""
    NORMAL = 'normal'
    HIGH_VOLATILITY = 'high_volatility'
    LOW_VOLATILITY = 'low_volatility'
    TRENDING_BULLISH = 'trending_bullish'
    TRENDING_BEARISH = 'trending_bearish'
    RANGING_NARROW = 'ranging_narrow'
    RANGING_WIDE = 'ranging_wide'
    BREAKOUT_BULLISH = 'breakout_bullish'
    BREAKOUT_BEARISH = 'breakout_bearish'
    REVERSAL_BULLISH = 'reversal_bullish'
    REVERSAL_BEARISH = 'reversal_bearish'
    LIQUIDITY_GAP = 'liquidity_gap'
    FLASH_CRASH = 'flash_crash'
    FLASH_SPIKE = 'flash_spike'
    NEWS_REACTION = 'news_reaction'


class MarketSession(str, Enum):
    """Trading sessions to model session-specific behavior."""
    SYDNEY = 'sydney'
    TOKYO = 'tokyo'
    LONDON = 'london'
    NEWYORK = 'newyork'
    OVERLAP_TOKYO_LONDON = 'overlap_tokyo_london'
    OVERLAP_LONDON_NEWYORK = 'overlap_london_newyork'


class LiquidityProfile(str, Enum):
    """Liquidity profiles affecting spread, slippage and execution."""
    VERY_HIGH = 'very_high'
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'
    VERY_LOW = 'very_low'


class SimulationScenario:
    """Defines a market simulation scenario with specific conditions."""

    def __init__(self, name: str, symbol: str, duration: timedelta,
        market_condition: MarketCondition, liquidity_profile:
        LiquidityProfile, volatility_factor: float=1.0, spread_factor:
        float=1.0, trend_strength: float=0.0, mean_reversion_strength:
        float=0.0, price_jump_probability: float=0.0, price_jump_magnitude:
        float=0.0, special_events: List[Dict[str, Any]]=None, description:
        str=''):
        """
        Initialize a simulation scenario.
        
        Args:
            name: Scenario name
            symbol: Trading symbol
            duration: Duration of the scenario
            market_condition: Primary market condition
            liquidity_profile: Liquidity characteristics
            volatility_factor: Multiplier for volatility (1.0 = normal)
            spread_factor: Multiplier for spread (1.0 = normal)
            trend_strength: Strength and direction of trend
            mean_reversion_strength: Strength of mean reversion
            price_jump_probability: Probability of price jumps per tick
            price_jump_magnitude: Average size of jumps as % of price
            special_events: List of special event configurations
            description: Description of the scenario
        """
        self.name = name
        self.symbol = symbol
        self.duration = duration
        self.market_condition = market_condition
        self.liquidity_profile = liquidity_profile
        self.volatility_factor = volatility_factor
        self.spread_factor = spread_factor
        self.trend_strength = trend_strength
        self.mean_reversion_strength = mean_reversion_strength
        self.price_jump_probability = price_jump_probability
        self.price_jump_magnitude = price_jump_magnitude
        self.special_events = special_events or []
        self.description = description

    def __str__(self) ->str:
        """Get string representation of the scenario."""
        return (
            f'Scenario: {self.name} ({self.symbol}, {self.duration}), Condition: {self.market_condition.value}, Liquidity: {self.liquidity_profile.value}'
            )


class AdvancedMarketRegimeSimulator:
    """
    Enhanced market simulation for realistic RL agent training.
    
    This simulator creates realistic forex market conditions with:
    - Fine-grained control over market regimes and conditions
    - Session-specific behavior modeling
    - Realistic spread and slippage models
    - Extreme events like flash crashes, gaps, and liquidity crunches
    - Integration with news and sentiment effects
    """

    def __init__(self, broker_simulator: ForexBrokerSimulator,
        news_simulator: Optional[NewsAndSentimentSimulator]=None,
        base_volatility_map: Optional[Dict[str, float]]=None,
        base_spread_map: Optional[Dict[str, float]]=None, random_seed:
        Optional[int]=None):
        """
        Initialize the advanced market regime simulator.
        
        Args:
            broker_simulator: Forex broker simulator to enhance
            news_simulator: Optional news simulator for event integration
            base_volatility_map: Base volatility levels by symbol
            base_spread_map: Base spread levels by symbol
            random_seed: Optional seed for reproducibility
        """
        self.broker_simulator = broker_simulator
        self.news_simulator = news_simulator
        self.base_volatility_map = base_volatility_map or {}
        self.base_spread_map = base_spread_map or {}
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        self.current_scenario = None
        self.active_conditions = {}
        self.session_schedule = self._create_default_session_schedule()
        self.historical_regimes = {}
        self.session_liquidity_map = {MarketSession.SYDNEY: 0.8,
            MarketSession.TOKYO: 0.9, MarketSession.LONDON: 1.2,
            MarketSession.NEWYORK: 1.2, MarketSession.OVERLAP_TOKYO_LONDON:
            1.1, MarketSession.OVERLAP_LONDON_NEWYORK: 1.3}

    def _create_default_session_schedule(self) ->Dict[MarketSession, Tuple[
        int, int]]:
        """
        Create default forex session schedule in UTC hours.
        
        Returns:
            Dictionary mapping sessions to (start_hour, end_hour)
        """
        return {MarketSession.SYDNEY: (20, 5), MarketSession.TOKYO: (0, 9),
            MarketSession.LONDON: (8, 17), MarketSession.NEWYORK: (13, 22),
            MarketSession.OVERLAP_TOKYO_LONDON: (8, 9), MarketSession.
            OVERLAP_LONDON_NEWYORK: (13, 17)}

    @with_broker_api_resilience('get_active_sessions')
    def get_active_sessions(self, timestamp: datetime) ->List[MarketSession]:
        """
        Get active trading sessions for a given timestamp.
        
        Args:
            timestamp: Datetime to check
            
        Returns:
            List of active sessions
        """
        hour = timestamp.hour
        active = []
        for session, (start, end) in self.session_schedule.items():
            if start < end:
                if start <= hour < end:
                    active.append(session)
            elif hour >= start or hour < end:
                active.append(session)
        return active

    @with_broker_api_resilience('get_liquidity_factor')
    def get_liquidity_factor(self, timestamp: datetime) ->float:
        """
        Calculate liquidity factor based on active sessions.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Liquidity factor (higher means more liquid)
        """
        active_sessions = self.get_active_sessions(timestamp)
        if not active_sessions:
            return 0.7
        return max(self.session_liquidity_map.get(session, 1.0) for session in
            active_sessions)

    def apply_scenario(self, scenario: SimulationScenario, start_time: datetime
        ) ->None:
        """
        Apply a simulation scenario to the broker simulator.
        
        Args:
            scenario: Scenario configuration
            start_time: Starting timestamp for the scenario
        """
        self.current_scenario = scenario
        symbol = scenario.symbol
        self.active_conditions[symbol] = scenario.market_condition
        regime = self._condition_to_regime(scenario.market_condition)
        self.broker_simulator.set_market_regime(regime)
        self.broker_simulator.set_volatility_factor(scenario.volatility_factor)
        impact_params = {'trend_strength': scenario.trend_strength,
            'mean_reversion': scenario.mean_reversion_strength,
            'jump_probability': scenario.price_jump_probability,
            'jump_magnitude': scenario.price_jump_magnitude}
        self._configure_price_impacts(symbol, impact_params)
        if scenario.special_events and self.news_simulator:
            for event_config in scenario.special_events:
                self._add_special_event(event_config, start_time)
        logger.info(
            f'Applied scenario: {scenario.name} starting at {start_time}')

    def _condition_to_regime(self, condition: MarketCondition
        ) ->MarketRegimeType:
        """Map detailed market condition to broader regime type."""
        if condition in [MarketCondition.TRENDING_BULLISH, MarketCondition.
            TRENDING_BEARISH]:
            return MarketRegimeType.TRENDING
        elif condition in [MarketCondition.RANGING_NARROW, MarketCondition.
            RANGING_WIDE]:
            return MarketRegimeType.RANGING
        elif condition in [MarketCondition.HIGH_VOLATILITY, MarketCondition
            .BREAKOUT_BULLISH, MarketCondition.BREAKOUT_BEARISH,
            MarketCondition.FLASH_CRASH, MarketCondition.FLASH_SPIKE]:
            return MarketRegimeType.VOLATILE
        else:
            return MarketRegimeType.NORMAL

    def _configure_price_impacts(self, symbol: str, params: Dict[str, float]
        ) ->None:
        """Configure price impact parameters for the broker simulator."""
        if hasattr(self.broker_simulator, 'set_trend_strength'):
            self.broker_simulator.set_trend_strength(params['trend_strength'])
        if hasattr(self.broker_simulator, 'set_mean_reversion'):
            self.broker_simulator.set_mean_reversion(params['mean_reversion'])
        if hasattr(self.broker_simulator, 'set_jump_parameters'):
            self.broker_simulator.set_jump_parameters(params[
                'jump_probability'], params['jump_magnitude'])

    def _add_special_event(self, event_config: Dict[str, Any], base_time:
        datetime) ->None:
        """Add a special event to the news simulator."""
        if not self.news_simulator:
            return
        if 'time_offset_minutes' in event_config:
            event_time = base_time + timedelta(minutes=event_config[
                'time_offset_minutes'])
        else:
            event_time = base_time + timedelta(minutes=random.randint(5, 
                int(self.current_scenario.duration.total_seconds() / 60) - 5))
        event = NewsEvent(event_id=
            f'scenario_{self.current_scenario.name}_{len(self.news_simulator.events)}'
            , event_type=event_config_manager.get('event_type'), impact_level=
            event_config_manager.get('impact_level', NewsImpactLevel.HIGH),
            timestamp=event_time, currencies_affected=[self.
            current_scenario.symbol], title=event_config.get('title',
            'Scenario Event'), description=event_config.get('description',
            ''), expected_value=event_config_manager.get('expected_value', 0),
            actual_value=event_config_manager.get('actual_value', 0),
            previous_value=event_config_manager.get('previous_value', 0),
            sentiment_impact=event_config.get('sentiment_impact',
            SentimentLevel.NEUTRAL), volatility_impact=event_config.get(
            'volatility_impact', 1.0), price_impact=event_config.get(
            'price_impact', 0.0), duration_minutes=event_config.get(
            'duration_minutes', 30))
        self.news_simulator.add_news_event(event)

    def generate_market_data(self, symbol: str, start_time: datetime,
        end_time: datetime, timeframe: str='1m', include_indicators: bool=True
        ) ->pd.DataFrame:
        """
        Generate synthetic market data for a specific scenario.
        
        Args:
            symbol: Trading symbol
            start_time: Start timestamp
            end_time: End timestamp
            timeframe: Data timeframe
            include_indicators: Whether to include technical indicators
            
        Returns:
            DataFrame with generated market data
        """
        if timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
            periods = int((end_time - start_time).total_seconds() / (
                minutes * 60)) + 1
        elif timeframe.endswith('h'):
            hours = int(timeframe[:-1])
            periods = int((end_time - start_time).total_seconds() / (hours *
                3600)) + 1
        elif timeframe.endswith('d'):
            days = int(timeframe[:-1])
            periods = int((end_time - start_time).total_seconds() / (days *
                86400)) + 1
        else:
            raise ValueError(f'Unsupported timeframe: {timeframe}')
        timestamps = [(start_time + timedelta(minutes=i * minutes)) for i in
            range(periods)]
        condition = self.active_conditions.get(symbol, MarketCondition.NORMAL)
        prices = self._generate_price_series(condition=condition, periods=
            periods, base_price=self._get_base_price(symbol), volatility=
            self._get_volatility(symbol), timestamps=timestamps)
        data = self._generate_ohlcv(prices, timestamps, symbol)
        if include_indicators:
            data = self._add_technical_indicators(data)
        return data

    @with_exception_handling
    def _get_base_price(self, symbol: str) ->float:
        """Get base price for a symbol."""
        try:
            current_price = self.broker_simulator.get_current_price(symbol)
            if current_price > 0:
                return current_price
        except:
            pass
        default_prices = {'EUR/USD': 1.1, 'GBP/USD': 1.25, 'USD/JPY': 110.0,
            'USD/CHF': 0.92, 'AUD/USD': 0.75, 'USD/CAD': 1.3, 'NZD/USD': 
            0.7, 'EUR/GBP': 0.85}
        return default_prices.get(symbol, 1.0)

    def _get_volatility(self, symbol: str) ->float:
        """Get base volatility for a symbol."""
        if symbol in self.base_volatility_map:
            return self.base_volatility_map[symbol]
        default_volatility = {'EUR/USD': 0.005, 'GBP/USD': 0.007, 'USD/JPY':
            0.006, 'USD/CHF': 0.006, 'AUD/USD': 0.008, 'USD/CAD': 0.006,
            'NZD/USD': 0.008, 'EUR/GBP': 0.005}
        return default_volatility.get(symbol, 0.006)

    def _generate_price_series(self, condition: MarketCondition, periods:
        int, base_price: float, volatility: float, timestamps: List[datetime]
        ) ->np.ndarray:
        """
        Generate synthetic price series based on market condition.
        
        Args:
            condition: Market condition
            periods: Number of periods to generate
            base_price: Starting price
            volatility: Base volatility
            timestamps: List of timestamps for this series
            
        Returns:
            NumPy array of price points
        """
        params = self._get_condition_parameters(condition)
        if len(timestamps) > 1:
            minutes_diff = (timestamps[1] - timestamps[0]).total_seconds() / 60
            timeframe_factor = np.sqrt(minutes_diff / (24 * 60))
            adjusted_volatility = volatility * timeframe_factor
        else:
            adjusted_volatility = volatility / np.sqrt(252 * 24 * 60)
        adjusted_volatility *= params['volatility_factor']
        prices = np.zeros(periods)
        prices[0] = base_price
        liquidity_factors = [self.get_liquidity_factor(ts) for ts in timestamps
            ]
        for i in range(1, periods):
            liquidity = liquidity_factors[i]
            drift = params['trend'] * base_price * 0.0001
            if params['mean_reversion'] > 0 and i > 1:
                lookback = min(20, i)
                mean_price = np.mean(prices[i - lookback:i])
                mean_reversion = params['mean_reversion'] * (mean_price -
                    prices[i - 1]) * 0.1
            else:
                mean_reversion = 0
            random_change = np.random.normal(0, adjusted_volatility / np.
                sqrt(liquidity))
            if np.random.random() < params['jump_probability']:
                jump_direction = 1 if np.random.random() > 0.5 else -1
                jump_size = np.random.exponential(params['jump_size']
                    ) * base_price * jump_direction
            else:
                jump_size = 0
            price_change = drift + mean_reversion + random_change * prices[
                i - 1] + jump_size
            prices[i] = prices[i - 1] + price_change
            prices[i] = max(prices[i], base_price * 0.5)
            prices[i] = min(prices[i], base_price * 1.5)
        return prices

    def _get_condition_parameters(self, condition: MarketCondition) ->Dict[
        str, float]:
        """Get simulation parameters for a specific market condition."""
        default_params = {'trend': 0.0, 'volatility_factor': 1.0,
            'mean_reversion': 0.0, 'jump_probability': 0.0, 'jump_size': 0.001}
        condition_params = {MarketCondition.NORMAL: {}, MarketCondition.
            HIGH_VOLATILITY: {'volatility_factor': 2.0, 'jump_probability':
            0.01}, MarketCondition.LOW_VOLATILITY: {'volatility_factor': 
            0.5, 'mean_reversion': 0.2}, MarketCondition.TRENDING_BULLISH:
            {'trend': 0.5, 'mean_reversion': -0.1}, MarketCondition.
            TRENDING_BEARISH: {'trend': -0.5, 'mean_reversion': -0.1},
            MarketCondition.RANGING_NARROW: {'volatility_factor': 0.7,
            'mean_reversion': 0.5}, MarketCondition.RANGING_WIDE: {
            'volatility_factor': 1.3, 'mean_reversion': 0.3},
            MarketCondition.BREAKOUT_BULLISH: {'trend': 0.8,
            'volatility_factor': 1.5, 'jump_probability': 0.05, 'jump_size':
            0.002}, MarketCondition.BREAKOUT_BEARISH: {'trend': -0.8,
            'volatility_factor': 1.5, 'jump_probability': 0.05, 'jump_size':
            0.002}, MarketCondition.REVERSAL_BULLISH: {'trend': 0.4,
            'volatility_factor': 1.3, 'jump_probability': 0.03},
            MarketCondition.REVERSAL_BEARISH: {'trend': -0.4,
            'volatility_factor': 1.3, 'jump_probability': 0.03},
            MarketCondition.LIQUIDITY_GAP: {'volatility_factor': 1.8,
            'jump_probability': 0.1, 'jump_size': 0.003}, MarketCondition.
            FLASH_CRASH: {'trend': -2.0, 'volatility_factor': 3.0,
            'jump_probability': 0.3, 'jump_size': 0.005}, MarketCondition.
            FLASH_SPIKE: {'trend': 2.0, 'volatility_factor': 3.0,
            'jump_probability': 0.3, 'jump_size': 0.005}, MarketCondition.
            NEWS_REACTION: {'volatility_factor': 2.0, 'jump_probability': 
            0.2, 'jump_size': 0.003}}
        params = default_params.copy()
        if condition in condition_params:
            params.update(condition_params[condition])
        return params

    def _generate_ohlcv(self, prices: np.ndarray, timestamps: List[datetime
        ], symbol: str) ->pd.DataFrame:
        """
        Generate OHLCV data from price series.
        
        Args:
            prices: NumPy array of prices
            timestamps: List of timestamps
            symbol: Trading symbol
            
        Returns:
            DataFrame with OHLCV data
        """
        data = pd.DataFrame(index=timestamps)
        n = len(prices)
        opens = prices.copy()
        closes = prices.copy()
        opens[1:] = opens[1:] * (1 + np.random.normal(0, 0.0002, n - 1))
        highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0,
            0.0005, n)))
        lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 
            0.0005, n)))
        volumes = np.abs(np.random.normal(1000, 300, n))
        price_changes = np.abs(np.diff(np.append(prices[0], prices)))
        volume_factors = 1 + 5 * price_changes / np.mean(price_changes)
        volumes = volumes * volume_factors
        base_spread = self.base_spread_map.get(symbol, self.
            _get_default_spread(symbol))
        liquidity_factors = np.array([self.get_liquidity_factor(ts) for ts in
            timestamps])
        volatility_factor = np.std(prices) / (prices[0] * 0.001)
        spreads = base_spread * (1.5 - 0.5 * liquidity_factors) * (0.8 + 
            0.4 * volatility_factor)
        data['open'] = opens
        data['high'] = highs
        data['low'] = lows
        data['close'] = closes
        data['volume'] = volumes.astype(int)
        data['spread'] = spreads
        return data

    def _get_default_spread(self, symbol: str) ->float:
        """Get default spread for a symbol in pips."""
        default_spreads = {'EUR/USD': 1.0, 'GBP/USD': 1.5, 'USD/JPY': 1.5,
            'USD/CHF': 2.0, 'AUD/USD': 1.5, 'USD/CAD': 2.0, 'NZD/USD': 2.0,
            'EUR/GBP': 1.8}
        return default_spreads.get(symbol, 2.5)

    def _add_technical_indicators(self, data: pd.DataFrame) ->pd.DataFrame:
        """Add technical indicators to the generated data."""
        df = data.copy()
        df['sma5'] = df['close'].rolling(window=5).mean()
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - 100 / (1 + rs)
        df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'
            ].ewm(span=26, adjust=False).mean()
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close}
            ).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        return df

    def generate_scenario_library(self) ->Dict[str, SimulationScenario]:
        """
        Generate a library of predefined simulation scenarios.
        
        Returns:
            Dictionary of named scenarios
        """
        scenarios = {}
        symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY']
        scenarios['normal_market'] = SimulationScenario(name=
            'normal_market', symbol='EUR/USD', duration=timedelta(days=1),
            market_condition=MarketCondition.NORMAL, liquidity_profile=
            LiquidityProfile.HIGH, description=
            'Standard market conditions with normal liquidity and volatility')
        scenarios['strong_uptrend'] = SimulationScenario(name=
            'strong_uptrend', symbol='EUR/USD', duration=timedelta(days=1),
            market_condition=MarketCondition.TRENDING_BULLISH,
            liquidity_profile=LiquidityProfile.HIGH, trend_strength=0.8,
            description='Strong bullish trend with momentum')
        scenarios['strong_downtrend'] = SimulationScenario(name=
            'strong_downtrend', symbol='EUR/USD', duration=timedelta(days=1
            ), market_condition=MarketCondition.TRENDING_BEARISH,
            liquidity_profile=LiquidityProfile.HIGH, trend_strength=-0.8,
            description='Strong bearish trend with momentum')
        scenarios['tight_range'] = SimulationScenario(name='tight_range',
            symbol='EUR/USD', duration=timedelta(days=1), market_condition=
            MarketCondition.RANGING_NARROW, liquidity_profile=
            LiquidityProfile.MEDIUM, volatility_factor=0.6,
            mean_reversion_strength=0.7, description=
            'Tight consolidation with strong mean reversion')
        scenarios['wide_range'] = SimulationScenario(name='wide_range',
            symbol='EUR/USD', duration=timedelta(days=1), market_condition=
            MarketCondition.RANGING_WIDE, liquidity_profile=
            LiquidityProfile.MEDIUM, volatility_factor=1.3,
            mean_reversion_strength=0.5, description=
            'Wide range consolidation with elevated volatility')
        scenarios['bullish_breakout'] = SimulationScenario(name=
            'bullish_breakout', symbol='EUR/USD', duration=timedelta(hours=
            12), market_condition=MarketCondition.BREAKOUT_BULLISH,
            liquidity_profile=LiquidityProfile.MEDIUM, volatility_factor=
            1.8, trend_strength=0.9, price_jump_probability=0.05,
            price_jump_magnitude=0.002, description=
            'Bullish breakout with increased volatility and volume')
        scenarios['high_volatility'] = SimulationScenario(name=
            'high_volatility', symbol='EUR/USD', duration=timedelta(hours=8
            ), market_condition=MarketCondition.HIGH_VOLATILITY,
            liquidity_profile=LiquidityProfile.MEDIUM, volatility_factor=
            2.5, price_jump_probability=0.03, price_jump_magnitude=0.001,
            description='Highly volatile market with erratic price movement')
        scenarios['nfp_release'] = SimulationScenario(name='nfp_release',
            symbol='EUR/USD', duration=timedelta(hours=4), market_condition
            =MarketCondition.NEWS_REACTION, liquidity_profile=
            LiquidityProfile.LOW, volatility_factor=2.5,
            price_jump_probability=0.2, price_jump_magnitude=0.003,
            special_events=[{'event_type': 'ECONOMIC_DATA', 'impact_level':
            NewsImpactLevel.HIGH, 'title': 'US Non-Farm Payrolls',
            'time_offset_minutes': 30, 'volatility_impact': 3.0,
            'price_impact': 0.005, 'duration_minutes': 60}], description=
            'High impact news event with significant price reaction')
        scenarios['low_liquidity'] = SimulationScenario(name=
            'low_liquidity', symbol='EUR/USD', duration=timedelta(hours=6),
            market_condition=MarketCondition.LIQUIDITY_GAP,
            liquidity_profile=LiquidityProfile.VERY_LOW, spread_factor=3.0,
            volatility_factor=1.5, price_jump_probability=0.08,
            price_jump_magnitude=0.002, description=
            'Very low liquidity condition with wide spreads and potential gaps'
            )
        scenarios['flash_crash'] = SimulationScenario(name='flash_crash',
            symbol='EUR/USD', duration=timedelta(hours=2), market_condition
            =MarketCondition.FLASH_CRASH, liquidity_profile=
            LiquidityProfile.VERY_LOW, volatility_factor=4.0, spread_factor
            =5.0, trend_strength=-2.0, price_jump_probability=0.3,
            price_jump_magnitude=0.01, description=
            'Sudden market crash with extreme volatility and poor liquidity')
        return scenarios

    @with_broker_api_resilience('get_random_scenario')
    def get_random_scenario(self, symbol: Optional[str]=None
        ) ->SimulationScenario:
        """
        Get a random scenario from the scenario library.
        
        Args:
            symbol: Optional symbol to filter scenarios by
            
        Returns:
            Random simulation scenario
        """
        scenarios = list(self.generate_scenario_library().values())
        if symbol:
            scenarios = [s for s in scenarios if s.symbol == symbol]
        if not scenarios:
            return SimulationScenario(name='default_scenario', symbol=
                symbol or 'EUR/USD', duration=timedelta(days=1),
                market_condition=MarketCondition.NORMAL, liquidity_profile=
                LiquidityProfile.HIGH)
        return random.choice(scenarios)

    @with_broker_api_resilience('create_curriculum')
    def create_curriculum(self, symbol: str, difficulty_levels: int=5,
        scenario_duration: timedelta=timedelta(hours=8)) ->Dict[int, List[
        SimulationScenario]]:
        """
        Create a curriculum of progressively challenging scenarios.
        
        Args:
            symbol: Trading symbol to create curriculum for
            difficulty_levels: Number of difficulty levels
            scenario_duration: Duration for each scenario
            
        Returns:
            Dictionary mapping difficulty levels to lists of scenarios
        """
        curriculum = {}
        curriculum[1] = [SimulationScenario(name='level1_normal', symbol=
            symbol, duration=scenario_duration, market_condition=
            MarketCondition.NORMAL, liquidity_profile=LiquidityProfile.HIGH,
            description='Level 1: Normal market conditions'),
            SimulationScenario(name='level1_mild_trend', symbol=symbol,
            duration=scenario_duration, market_condition=MarketCondition.
            TRENDING_BULLISH, liquidity_profile=LiquidityProfile.HIGH,
            trend_strength=0.3, description='Level 1: Mild bullish trend')]
        curriculum[2] = [SimulationScenario(name='level2_ranging', symbol=
            symbol, duration=scenario_duration, market_condition=
            MarketCondition.RANGING_WIDE, liquidity_profile=
            LiquidityProfile.MEDIUM, mean_reversion_strength=0.4,
            description='Level 2: Wide ranging market'), SimulationScenario
            (name='level2_reversal', symbol=symbol, duration=
            scenario_duration, market_condition=MarketCondition.
            REVERSAL_BULLISH, liquidity_profile=LiquidityProfile.MEDIUM,
            volatility_factor=1.2, description=
            'Level 2: Market reversal with increased volatility')]
        curriculum[3] = [SimulationScenario(name='level3_volatility',
            symbol=symbol, duration=scenario_duration, market_condition=
            MarketCondition.HIGH_VOLATILITY, liquidity_profile=
            LiquidityProfile.MEDIUM, volatility_factor=1.8, description=
            'Level 3: Higher volatility market'), SimulationScenario(name=
            'level3_breakout', symbol=symbol, duration=scenario_duration,
            market_condition=MarketCondition.BREAKOUT_BULLISH,
            liquidity_profile=LiquidityProfile.MEDIUM, trend_strength=0.7,
            price_jump_probability=0.03, description=
            'Level 3: Bullish breakout'), SimulationScenario(name=
            'level3_news', symbol=symbol, duration=scenario_duration,
            market_condition=MarketCondition.NEWS_REACTION,
            liquidity_profile=LiquidityProfile.MEDIUM, volatility_factor=
            1.5, special_events=[{'event_type': 'ECONOMIC_DATA',
            'impact_level': NewsImpactLevel.MEDIUM, 'volatility_impact': 
            1.5, 'price_impact': 0.002}], description=
            'Level 3: Medium-impact news event')]
        curriculum[4] = [SimulationScenario(name='level4_liquidity', symbol
            =symbol, duration=scenario_duration, market_condition=
            MarketCondition.LIQUIDITY_GAP, liquidity_profile=
            LiquidityProfile.LOW, spread_factor=2.0, price_jump_probability
            =0.05, description='Level 4: Low liquidity conditions'),
            SimulationScenario(name='level4_major_news', symbol=symbol,
            duration=scenario_duration, market_condition=MarketCondition.
            NEWS_REACTION, liquidity_profile=LiquidityProfile.LOW,
            volatility_factor=2.0, special_events=[{'event_type':
            'ECONOMIC_DATA', 'impact_level': NewsImpactLevel.HIGH,
            'volatility_impact': 2.5, 'price_impact': 0.004}], description=
            'Level 4: High-impact news event'), SimulationScenario(name=
            'level4_trend_change', symbol=symbol, duration=
            scenario_duration, market_condition=MarketCondition.
            TRENDING_BULLISH, liquidity_profile=LiquidityProfile.MEDIUM,
            trend_strength=0.8, special_events=[{'event_type':
            'MARKET_STRUCTURE', 'time_offset_minutes': int(
            scenario_duration.total_seconds() / 60 / 2), 'description':
            'Sudden trend reversal', 'price_impact': -0.005}], description=
            'Level 4: Trend with sudden reversal')]
        curriculum[5] = [SimulationScenario(name='level5_flash_crash',
            symbol=symbol, duration=scenario_duration, market_condition=
            MarketCondition.FLASH_CRASH, liquidity_profile=LiquidityProfile
            .VERY_LOW, volatility_factor=3.0, spread_factor=3.5,
            price_jump_probability=0.2, price_jump_magnitude=0.008,
            description='Level 5: Flash crash with extreme volatility'),
            SimulationScenario(name='level5_complex', symbol=symbol,
            duration=scenario_duration, market_condition=MarketCondition.
            HIGH_VOLATILITY, liquidity_profile=LiquidityProfile.LOW,
            volatility_factor=2.5, special_events=[{'event_type':
            'ECONOMIC_DATA', 'impact_level': NewsImpactLevel.HIGH,
            'time_offset_minutes': int(scenario_duration.total_seconds() / 
            60 / 3), 'volatility_impact': 3.0, 'price_impact': 0.006}, {
            'event_type': 'CENTRAL_BANK', 'impact_level': NewsImpactLevel.
            HIGH, 'time_offset_minutes': int(scenario_duration.
            total_seconds() / 60 * 2 / 3), 'volatility_impact': 2.5,
            'price_impact': -0.008}], description=
            'Level 5: Complex scenario with multiple high-impact events'),
            SimulationScenario(name='level5_extreme_volatility', symbol=
            symbol, duration=scenario_duration, market_condition=
            MarketCondition.HIGH_VOLATILITY, liquidity_profile=
            LiquidityProfile.VERY_LOW, volatility_factor=4.0, spread_factor
            =4.0, price_jump_probability=0.15, price_jump_magnitude=0.01,
            description='Level 5: Extreme volatility with poor liquidity')]
        return curriculum
