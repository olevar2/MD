"""
Integration tests for news/sentiment and RL environment integration.

These tests validate that the news and sentiment simulator properly
affects the RL environment and that the agent can learn to adapt to
news events appropriately.
"""

import unittest
import numpy as np
from datetime import datetime, timedelta

from trading_gateway_service.simulation.forex_broker_simulator import ForexBrokerSimulator, OrderSide
from trading_gateway_service.simulation.news_sentiment_simulator import (
    NewsAndSentimentSimulator, NewsEvent, NewsEventType, NewsImpactLevel, SentimentLevel
)
from core.enhanced_rl_env import EnhancedForexTradingEnv

class TestNewsRLIntegration(unittest.TestCase):
    """Test integration between news/sentiment simulator and RL environment."""
    
    def setUp(self):
        """Set up test environment."""
        # Create broker simulator
        self.broker_simulator = ForexBrokerSimulator(
            balance=10000.0,
            leverage=100,
            symbols=["EUR/USD"],
            fee_percent=0.001
        )
        
        # Create news simulator with fixed seed for reproducibility
        self.news_simulator = NewsAndSentimentSimulator(seed=42)
        
        # Generate a test calendar
        start_date = datetime.now()
        end_date = start_date + timedelta(days=1)
        self.news_simulator.generate_random_economic_calendar(
            start_date, end_date, ["EUR/USD"], num_events=5
        )
        
        # Create RL environment
        self.env = EnhancedForexTradingEnv(
            broker_simulator=self.broker_simulator,
            symbol="EUR/USD",
            timeframes=["1m", "5m"],
            lookback_periods=10,
            include_news_sentiment=True,
            news_sentiment_simulator=self.news_simulator
        )
    
    def test_news_features_in_observation_space(self):
        """Test that news features are included in the observation space."""
        observation = self.env.reset()
        
        # Verify observation dimensions include news component
        self.assertIn("news", self.env._observation_dimensions)
        self.assertGreater(self.env._observation_dimensions["news"], 0)
        
        # Verify total observation size
        expected_size = sum(dim for name, dim in self.env._observation_dimensions.items())
        self.assertEqual(len(observation), expected_size)
    
    def test_news_event_affects_observation(self):
        """Test that a news event affects the observation."""
        # Reset environment to start fresh
        self.env.reset()
        
        # Get base observation without active news
        base_observation = self.env._get_observation()
        
        # Create and add a significant news event
        event = NewsEvent(
            event_id="test_event",
            event_type=NewsEventType.ECONOMIC_DATA,
            impact_level=NewsImpactLevel.HIGH,
            timestamp=datetime.now(),
            currencies_affected=["EUR/USD"],
            title="Test NFP Release",
            price_impact=0.005,  # 0.5% price impact
            volatility_impact=2.0  # Doubles volatility
        )
        self.news_simulator.add_news_event(event)
        
        # Get observation with active news
        news_observation = self.env._get_observation()
        
        # Extract news features
        news_dim = self.env._observation_dimensions["news"]
        if news_dim > 0:
            news_features_index_start = sum(self.env._observation_dimensions[k] for k in 
                                          ["base", "technical", "order_book", "broker"])
            news_features = news_observation[news_features_index_start:news_features_index_start+news_dim]
            
            # Verify that some news features are non-zero
            self.assertTrue(np.any(news_features != 0), "News features should change with active news events")
    
    def test_news_affects_reward(self):
        """Test that news events affect the reward components."""
        # Reset environment
        self.env.reset()
        
        # Add a high-impact news event
        event = NewsEvent(
            event_id="test_event",
            event_type=NewsEventType.CENTRAL_BANK,
            impact_level=NewsImpactLevel.CRITICAL,
            timestamp=datetime.now(),
            currencies_affected=["EUR/USD"],
            title="Interest Rate Decision",
            sentiment_impact=SentimentLevel.VERY_BULLISH,
            price_impact=0.01,  # 1% price impact
            volatility_impact=3.0  # Triples volatility
        )
        self.news_simulator.add_news_event(event)
        
        # Take a buy action during bullish news
        # [action_type=BUY, size=0.5, sl_pips=20, tp_pips=40]
        aligned_action = np.array([1, 0.5, 20.0, 40.0])
        _, reward1, _, _ = self.env.step(aligned_action)
        
        # Reset and do the same test with a contradictory action
        self.env.reset()
        self.news_simulator.add_news_event(event)
        
        # Take a sell action during bullish news (contradictory)
        # [action_type=SELL, size=0.5, sl_pips=20, tp_pips=40]
        contradictory_action = np.array([2, 0.5, 20.0, 40.0])
        _, reward2, _, _ = self.env.step(contradictory_action)
        
        # The aligned action should receive a higher reward
        self.assertGreater(reward1, reward2, 
                         "Actions aligned with news sentiment should receive higher rewards")
    
    def test_sync_time_between_simulators(self):
        """Test that time synchronization works between simulators."""
        self.env.reset()
        
        # Record initial time
        initial_time = self.broker_simulator.current_time
        
        # Take a few steps
        for _ in range(5):
            action = np.array([0, 0, 0.0, 0.0])  # HOLD action
            self.env.step(action)
        
        # Verify both simulators have advanced time
        self.assertNotEqual(self.broker_simulator.current_time, initial_time)
        self.assertEqual(self.broker_simulator.current_time, self.news_simulator.current_time)

if __name__ == "__main__":
    unittest.main()
