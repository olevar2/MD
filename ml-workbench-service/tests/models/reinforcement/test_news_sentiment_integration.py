"""
Tests for the integration of NewsAndSentimentSimulator with EnhancedForexTradingEnv.

These tests verify that:
1. News events are correctly incorporated in the state/observation
2. News affects rewards appropriately 
3. Time synchronization works properly
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

from ml_workbench_service.models.reinforcement.enhanced_rl_env import (
    EnhancedForexTradingEnv, RewardComponent
)
from trading_gateway_service.simulation.forex_broker_simulator import (
    ForexBrokerSimulator, OrderSide, OrderType, MarketRegimeType
)
from trading_gateway_service.simulation.news_sentiment_simulator import (
    NewsAndSentimentSimulator, NewsEvent, NewsImpactLevel, SentimentLevel, NewsEventType
)
from core_foundations.models.financial_instruments import SymbolInfo


class MockForexBrokerSimulator:
    """Mock ForexBrokerSimulator for testing."""
    def __init__(self, symbol="EUR/USD", initial_balance=10000):
    """
      init  .
    
    Args:
        symbol: Description of symbol
        initial_balance: Description of initial_balance
    
    """

        self.symbol = symbol
        self.balance = initial_balance
        self.current_time = datetime.now()
        self.positions = []
        self.orders = []
        self.prices = {}
        
        # Generate some sample price data
        start_price = 1.1000
        self.price_data = [start_price]
        for _ in range(1000):
            prev_price = self.price_data[-1]
            change = random.uniform(-0.0005, 0.0005)
            self.price_data.append(prev_price + change)
    
    def get_historical_data(self, symbol, timeframe, periods, end_time=None):
        """Get mock historical data."""
        end_idx = min(len(self.price_data), periods)
        data = {
            'open': self.price_data[:end_idx],
            'high': [p + random.uniform(0.0001, 0.0003) for p in self.price_data[:end_idx]],
            'low': [p - random.uniform(0.0001, 0.0003) for p in self.price_data[:end_idx]],
            'close': self.price_data[:end_idx],
            'volume': [random.uniform(100, 1000) for _ in range(end_idx)],
            'spread': [random.uniform(0.0001, 0.0003) for _ in range(end_idx)]
        }
        dates = [self.current_time - timedelta(minutes=i) for i in range(end_idx)]
        dates.reverse()
        return pd.DataFrame(data, index=dates)
    
    def get_current_price(self, symbol, timestamp=None):
        """Get mock current price."""
        return self.price_data[-1]
    
    def get_price_change(self, symbol, from_time, to_time):
        """Get mock price change."""
        return random.uniform(-0.0005, 0.0005)
    
    def set_current_time(self, time):
        """Set current time."""
        self.current_time = time
    
    def get_available_times(self, symbol):
        """Get available timestamps."""
        now = datetime.now()
        return [now - timedelta(minutes=i) for i in range(1000)]
    
    def execute_order(self, symbol, side, order_type, size, timestamp):
        """Execute a mock order."""
        price = self.get_current_price(symbol)
        fee = size * 0.0001
        return {
            'order_id': len(self.orders) + 1,
            'symbol': symbol,
            'side': side,
            'order_type': order_type,
            'size': size,
            'avg_price': price,
            'fee': fee,
            'realized_pnl': random.uniform(-50, 50),  # Random P&L for testing
            'status': 'filled'
        }
    
    def get_order_book(self, symbol, timestamp=None):
        """Get mock order book."""
        current_price = self.get_current_price(symbol)
        return {
            'bids': [(current_price - 0.0001 * i, random.uniform(1, 10)) for i in range(1, 6)],
            'asks': [(current_price + 0.0001 * i, random.uniform(1, 10)) for i in range(1, 6)]
        }
    
    def set_market_regime(self, regime):
        """Set market regime (mock)."""
        pass
    
    def set_volatility_factor(self, factor):
        """Set volatility factor (mock)."""
        pass
    
    def enable_extreme_events(self, enable):
        """Enable/disable extreme events (mock)."""
        pass


@pytest.fixture
def mock_data():
    """Create mock price data for testing."""
    # Generate sample price data
    n = 100
    dates = [datetime.now() - timedelta(minutes=i) for i in range(n)]
    dates.reverse()
    
    # Generate price data
    base_price = 1.1000
    data = {
        'open': [base_price],
        'high': [base_price],
        'low': [base_price],
        'close': [base_price],
        'volume': [1000],
        'spread': [0.0002]
    }
    
    # Generate random walk
    for i in range(1, n):
        change = np.random.normal(0, 0.0005)
        new_price = data['close'][-1] + change
        data['open'].append(new_price)
        data['close'].append(new_price + np.random.normal(0, 0.0001))
        data['high'].append(max(data['open'][i], data['close'][i]) + abs(np.random.normal(0, 0.0002)))
        data['low'].append(min(data['open'][i], data['close'][i]) - abs(np.random.normal(0, 0.0002)))
        data['volume'].append(abs(np.random.normal(1000, 200)))
        data['spread'].append(abs(np.random.normal(0.0002, 0.00005)))
    
    df = pd.DataFrame(data, index=dates)
    
    # Add some technical indicators
    df['rsi'] = np.random.normal(50, 10, n)
    df['macd'] = np.random.normal(0, 0.0002, n)
    df['bb_upper'] = df['close'] + 0.002
    df['bb_lower'] = df['close'] - 0.002
    
    return df


@pytest.fixture
def news_events(mock_data):
    """Create mock news events for testing."""
    start_time = mock_data.index[0]
    mid_time = mock_data.index[len(mock_data) // 2]
    end_time = mock_data.index[-1]
    
    return [
        NewsEvent(
            event_id="ecb_rate_1",
            event_type=NewsEventType.CENTRAL_BANK,
            impact_level=NewsImpactLevel.HIGH,
            timestamp=mid_time - timedelta(hours=1),
            currencies_affected=["EUR/USD"],
            title="ECB Interest Rate Decision",
            description="ECB maintains interest rates",
            expected_value=0.5,
            actual_value=0.5,
            previous_value=0.25,
            sentiment_impact=SentimentLevel.NEUTRAL,
            volatility_impact=2.0,
            price_impact=0.0,
            duration_minutes=120
        ),
        NewsEvent(
            event_id="us_gdp_1",
            event_type=NewsEventType.ECONOMIC_DATA,
            impact_level=NewsImpactLevel.MEDIUM,
            timestamp=mid_time + timedelta(hours=2),
            currencies_affected=["EUR/USD", "USD/JPY"],
            title="US GDP",
            description="US GDP better than expected",
            expected_value=2.0,
            actual_value=2.1,
            previous_value=1.9,
            sentiment_impact=SentimentLevel.SLIGHTLY_BULLISH,
            volatility_impact=1.5,
            price_impact=0.001,
            duration_minutes=90
        ),
        NewsEvent(
            event_id="eu_inflation_1",
            event_type=NewsEventType.ECONOMIC_DATA,
            impact_level=NewsImpactLevel.HIGH,
            timestamp=end_time - timedelta(hours=5),
            currencies_affected=["EUR/USD", "EUR/GBP"],
            title="EU Inflation Rate",
            description="Eurozone inflation comes in higher than forecast",
            expected_value=1.8,
            actual_value=2.0,
            previous_value=1.7,
            sentiment_impact=SentimentLevel.BEARISH,
            volatility_impact=2.2,
            price_impact=-0.002,
            duration_minutes=150
        )
    ]


@pytest.fixture
def news_simulator(news_events):
    """Create a news and sentiment simulator with predefined events."""
    simulator = NewsAndSentimentSimulator(seed=42)
    
    # Add events to the simulator
    for event in news_events:
        simulator.add_news_event(event)
    
    # Set some currency sentiments
    simulator.set_sentiment("EUR", SentimentLevel.SLIGHTLY_BULLISH)
    simulator.set_sentiment("USD", SentimentLevel.NEUTRAL)
    
    return simulator


@pytest.fixture
def broker_simulator():
    """Create a mock broker simulator."""
    return MockForexBrokerSimulator(symbol="EUR/USD", initial_balance=10000)


@pytest.fixture
def env_with_news(mock_data, broker_simulator, news_simulator):
    """Create an environment with news and sentiment simulation enabled."""
    return EnhancedForexTradingEnv(
        broker_simulator=broker_simulator,
        symbol="EUR/USD",
        timeframes=["1m"],
        lookback_periods=10,
        features=["open", "high", "low", "close", "volume", "spread"],
        position_sizing_type="fixed",
        max_position_size=1.0,
        trading_fee_percent=0.002,
        reward_mode="risk_adjusted",
        risk_free_rate=0.02,
        episode_timesteps=100,
        time_step_seconds=60,
        random_episode_start=False,
        curriculum_level=0,
        include_broker_state=True,
        include_order_book=True,
        include_technical_indicators=True,
        include_news_sentiment=True,
        news_sentiment_simulator=news_simulator
    )


@pytest.fixture
def env_without_news(mock_data, broker_simulator):
    """Create an environment without news and sentiment simulation."""
    return EnhancedForexTradingEnv(
        broker_simulator=broker_simulator,
        symbol="EUR/USD",
        timeframes=["1m"],
        lookback_periods=10,
        features=["open", "high", "low", "close", "volume", "spread"],
        position_sizing_type="fixed",
        max_position_size=1.0,
        trading_fee_percent=0.002,
        reward_mode="risk_adjusted",
        risk_free_rate=0.02,
        episode_timesteps=100,
        time_step_seconds=60,
        random_episode_start=False,
        curriculum_level=0,
        include_broker_state=True,
        include_order_book=True,
        include_technical_indicators=True,
        include_news_sentiment=False,
        news_sentiment_simulator=None
    )


def test_environment_initialization(env_with_news, env_without_news):
    """Test that environments initialize correctly with and without news."""
    # Both environments should reset correctly
    obs_with_news = env_with_news.reset()
    obs_without_news = env_without_news.reset()
    
    # Environment with news should have larger observation space
    assert len(obs_with_news) >= len(obs_without_news)
    
    # Both should provide valid observations
    assert not np.any(np.isnan(obs_with_news))
    assert not np.any(np.isnan(obs_without_news))


def test_news_features_in_observation(env_with_news, news_simulator, news_events):
    """Test that news features are correctly included in the observation."""
    # Reset environment
    observation = env_with_news.reset()
    
    # Set environment time to just before a news event
    event_time = news_events[0].timestamp - timedelta(minutes=5)
    env_with_news.broker_simulator.set_time(event_time)
    news_simulator.set_current_time(event_time)
    
    # Get observation
    observation, _, _, _ = env_with_news.step([0])  # Hold action
    
    # Extract news features from observation
    # This would need to match the feature extraction logic in the environment
    # Since we don't have direct access to the feature indices, we'll check that
    # the observation changes when we advance time past the event
    
    observation_before = observation.copy()
    
    # Advance to after the event
    event_time = news_events[0].timestamp + timedelta(minutes=5)
    env_with_news.broker_simulator.set_time(event_time)
    news_simulator.set_current_time(event_time)
    
    observation_after, _, _, _ = env_with_news.step([0])  # Hold action
    
    # The observations should be different due to the news event
    assert not np.array_equal(observation_before, observation_after)


def test_news_adaptation_reward(env_with_news, news_simulator, news_events):
    """Test that the reward includes a bonus for adapting to news events."""
    # Reset environment
    env_with_news.reset()
    
    # Set environment time to just before a high-impact news event
    event_time = news_events[0].timestamp - timedelta(minutes=5)
    env_with_news.broker_simulator.set_time(event_time)
    news_simulator.set_current_time(event_time)
    
    # Take a step with a hold action during news
    _, reward_hold, _, info_hold = env_with_news.step([0])
    
    # Check reward components
    assert "reward_components" in info_hold
    assert "news_adaptation_bonus" in info_hold["reward_components"]
    
    # Take a step with a trade action during news
    _, reward_trade, _, info_trade = env_with_news.step([0.5])  # Buy position
    
    # Trade reward should be different due to news adaptation component
    assert reward_hold != reward_trade


def test_time_synchronization(env_with_news):
    """Test that the news simulator time is synchronized with the broker simulator."""
    # Reset environment
    env_with_news.reset()
    
    # Get initial time
    initial_time = env_with_news.broker_simulator.current_time
    
    # Take a step
    env_with_news.step([0])  # Hold action
    
    # Check that news simulator time advanced
    assert env_with_news.news_sentiment_simulator.current_time > initial_time
    
    # Check that times are synchronized
    assert env_with_news.news_sentiment_simulator.current_time == env_with_news.broker_simulator.current_time


def test_sentiment_changes_affect_observation(env_with_news, news_simulator):
    """Test that sentiment changes are reflected in the observation."""
    # Reset environment
    env_with_news.reset()
    
    # Record observation with neutral sentiment
    observation_neutral = env_with_news.reset()
    
    # Set positive sentiment
    current_time = env_with_news.broker_simulator.current_time
    news_simulator.set_sentiment("EUR", SentimentLevel.BULLISH)
    news_simulator.set_current_time(current_time)
    
    # Take a step to update observation
    observation_positive, _, _, _ = env_with_news.step([0])
    
    # Set negative sentiment
    news_simulator.set_sentiment("EUR", SentimentLevel.BEARISH)
    news_simulator.set_current_time(current_time)
    
    # Take a step to update observation
    observation_negative, _, _, _ = env_with_news.step([0])
    
    # Observations should differ due to sentiment changes
    assert not np.array_equal(observation_neutral, observation_positive)
    assert not np.array_equal(observation_positive, observation_negative)


if __name__ == "__main__":
    # Run tests
    pytest.main(["-xvs", __file__])
