"""
Test rl env integration module.

This module provides functionality for...
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Assuming paths are correctly set for imports
from core.enhanced_rl_env import EnhancedForexTradingEnv
from adapters.simulation_adapters import (
    BrokerSimulatorAdapter, MarketRegimeSimulatorAdapter
)
from common_lib.simulation.interfaces import (
    IBrokerSimulator, IMarketRegimeSimulator, MarketRegimeType
)

# Import for type hints only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from trading_gateway_service.simulation.forex_broker_simulator import ForexBrokerSimulator
    from trading_gateway_service.simulation.news_sentiment_simulator import (
        NewsAndSentimentSimulator, NewsEvent, NewsImpactLevel, SentimentLevel
    )
from core_foundations.models.financial_instruments import SymbolInfo

# Mock Data and Simulators
@pytest.fixture
def mock_symbol_info():
    """
    Mock symbol info.
    
    """

    return SymbolInfo(
        symbol="EURUSD",
        base_currency="EUR",
        quote_currency="USD",
        lot_size=100000,
        min_volume=0.01,
        max_volume=100,
        volume_step=0.01,
        pip_decimal_places=5,
        margin_rate=0.0333 # Approx 30:1 leverage
    )

@pytest.fixture
def mock_forex_data():
    """
    Mock forex data.
    
    """

    dates = pd.to_datetime([datetime(2023, 1, 1, i) for i in range(100)])
    data = {
        'open': np.random.rand(100) * 0.1 + 1.1,
        'high': lambda df: df['open'] + np.random.rand(100) * 0.01,
        'low': lambda df: df['open'] - np.random.rand(100) * 0.01,
        'close': lambda df: df['open'] + (np.random.rand(100) - 0.5) * 0.01,
        'volume': np.random.randint(100, 1000, 100),
        'rsi': np.random.rand(100) * 100,
        'macd': np.random.rand(100) * 0.01 - 0.005,
        'bb_upper': lambda df: df['close'] + 0.005,
        'bb_lower': lambda df: df['close'] - 0.005,
    }
    df = pd.DataFrame(data)
    # Evaluate lambda functions
    for col, val in data.items():
        if callable(val):
            df[col] = val(df)
    df.index = dates
    return df

@pytest.fixture
def broker_simulator(mock_forex_data, mock_symbol_info):
    """
    Broker simulator.
    
    Args:
        mock_forex_data: Description of mock_forex_data
        mock_symbol_info: Description of mock_symbol_info
    
    """

    # Use the adapter instead of direct dependency
    return BrokerSimulatorAdapter()

@pytest.fixture
def news_sentiment_simulator(mock_forex_data):
    """
    News sentiment simulator.
    
    Args:
        mock_forex_data: Description of mock_forex_data
    
    """

    # Use a simple mock object instead of direct dependency
    class MockNewsSentimentSimulator:
    """
    MockNewsSentimentSimulator class.
    
    Attributes:
        Add attributes here
    """

        def __init__(self):
    """
      init  .
    
    """

            start_time = mock_forex_data.index[0]
            event_time = start_time + timedelta(hours=10)
            self.events = [
                {
                    "timestamp": event_time,
                    "currency": "EUR",
                    "event_name": "ECB Rate Decision",
                    "impact": 3,  # HIGH impact
                    "actual": 0.5,
                    "forecast": 0.25,
                    "previous": 0.25
                }
            ]
            self.sentiment_over_time = {}

    return MockNewsSentimentSimulator()

@pytest.fixture
def integrated_rl_env(mock_forex_data, broker_simulator, news_sentiment_simulator):
    """
    Integrated rl env.
    
    Args:
        mock_forex_data: Description of mock_forex_data
        broker_simulator: Description of broker_simulator
        news_sentiment_simulator: Description of news_sentiment_simulator
    
    """

    return EnhancedForexTradingEnv(
        broker_simulator=broker_simulator,
        symbol="EUR/USD",
        timeframes=["1m"],
        lookback_periods=5,  # Keep small for testing
        features=["close", "volume", "rsi", "macd", "bb_upper", "bb_lower"],
        position_sizing_type="fixed",
        max_position_size=1.0,
        trading_fee_percent=0.002,
        reward_mode="risk_adjusted",
        episode_timesteps=50,
        include_news_sentiment=True,
        news_sentiment_simulator=news_sentiment_simulator,
        observation_normalization=False  # Simplify for testing
    )

# Test Cases
def test_env_initialization_with_simulators(integrated_rl_env):
    """Test if the environment initializes correctly with both simulators."""
    assert integrated_rl_env.broker_simulator is not None
    assert integrated_rl_env.news_sentiment_simulator is not None
    obs, info = integrated_rl_env.reset()
    assert isinstance(obs, np.ndarray)
    # Check if observation space includes the 3 news/sentiment features
    # Base features (5 lookback * 6 features) + Account (5) + Position (4) + News/Sentiment (3)
    # Note: The _get_observation_space flattens basedata, need to adjust expected shape
    # flat_base_data = 30, account = 5, position = 4, news/sentiment = 3 => Total = 42
    expected_obs_len = (5 * 6) + 5 + 4 + 3 # Based on current _get_state structure before potential flattening issues
    # Adjusting based on the current _get_state implementation which concatenates flattened base data
    expected_obs_len_flat = (5 * 6) + 5 + 4 + 0 + 3 # order_book_depth=0
    assert obs.shape == (expected_obs_len_flat,), f"Expected shape {(expected_obs_len_flat,)}, got {obs.shape}"


def test_news_event_impact_on_observation(integrated_rl_env):
    """Test if upcoming news event details appear in the observation state."""
    env = integrated_rl_env
    env.reset()

    # Find the step corresponding to the news event time
    news_event_time = env.news_sentiment_simulator.events[0].timestamp
    event_step_index = env.df.index.get_loc(news_event_time, method='nearest')

    # Step until just before the event
    steps_to_take = event_step_index - env.lookback_window - 5 # Step close to the event
    if steps_to_take <= 0:
         pytest.skip("Test setup issue: Event too early for lookback window.")

    for _ in range(steps_to_take):
        obs, _, _, _, _ = env.step(0) # Hold action

    # Check observation just before the event
    obs, _, _, _, _ = env.step(0) # Hold action
    news_features = obs[-3:] # Last 3 features are news/sentiment

    assert news_features[0] > 0 # Event impact should be non-zero (HIGH = 3)
    assert 0 < news_features[1] < 1 # Time to event should be small positive
    # Sentiment might still be neutral
    # print(f"Obs before event: {news_features}")

def test_trading_penalty_during_high_impact_news(integrated_rl_env):
    """Test if a penalty is applied when trading during a high-impact event."""
    env = integrated_rl_env
    env.reset()

    # Find the step corresponding to the news event time
    news_event_time = env.news_sentiment_simulator.events[0].timestamp
    event_step_index = env.df.index.get_loc(news_event_time, method='bfill') # Find step at or after event

    # Step until the event occurs
    steps_to_take = event_step_index - env.current_step
    if steps_to_take < 0:
         pytest.skip("Test setup issue: Event time calculation error.")

    for _ in range(steps_to_take):
         env.step(0) # Hold action

    # Now, take a trade action (Buy=1) exactly when the event is active
    _, reward_trade, _, _, _ = env.step(1) # Buy action

    # Take a hold action in the next step for comparison (assuming event duration is short)
    env.step(0) # Advance time past the event potentially
    _, reward_hold, _, _, _ = env.step(0) # Hold action

    print(f"Reward (Trade during event): {reward_trade}")
    print(f"Reward (Hold after event): {reward_hold}")

    # Expect the reward during the trade to be significantly lower due to the penalty
    # The exact reward depends on profit/loss, but the penalty should be applied.
    # Check if the news_penalty component was applied
    # This requires inspecting the reward calculation breakdown, which isn't directly returned.
    # We rely on the large negative penalty value defined in the fixture.
    assert reward_trade < reward_hold # Simplified check, assumes profit/loss is smaller than penalty
    assert reward_trade < -5.0 # Check if penalty was likely applied (given penalty is -10)


def test_sentiment_impact_on_observation(integrated_rl_env):
    """Test if sentiment changes are reflected in the observation."""
    env = integrated_rl_env
    env.reset()

    # Manually set sentiment in the simulator
    sentiment_time = env.current_time + timedelta(hours=2)
    env.news_sentiment_simulator.sentiment_over_time[sentiment_time] = SentimentLevel.POSITIVE

    # Step until sentiment change time
    steps_to_take = env.df.index.get_loc(sentiment_time, method='nearest') - env.current_step
    if steps_to_take <= 0:
         pytest.skip("Test setup issue: Sentiment time calculation error.")

    for _ in range(steps_to_take):
        obs, _, _, _, _ = env.step(0) # Hold

    # Check observation after sentiment change
    obs, _, _, _, _ = env.step(0) # Hold
    news_features = obs[-3:]

    assert news_features[2] == SentimentLevel.POSITIVE.value # Check sentiment value
    # print(f"Obs after sentiment change: {news_features}")

# Add more tests:
# - Test interaction with different news impact levels
# - Test interaction with different sentiment levels
# - Test edge cases (e.g., episode end during news event)
# - Test reset functionality clears news/sentiment state correctly
