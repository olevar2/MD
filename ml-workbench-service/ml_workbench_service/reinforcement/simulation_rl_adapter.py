"""
Simulation RL Adapter

This module connects the forex broker simulator and market regime simulator
with the RL environment to provide a unified interface for training agents
in realistic market conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta

from core_foundations.utils.logger import get_logger
from trading_gateway_service.simulation.forex_broker_simulator import (
    ForexBrokerSimulator, MarketRegimeType, LiquidityLevel, MarketEventType
)
from ml_workbench_service.models.reinforcement.rl_agent import ForexTradingEnvironment

logger = get_logger(__name__)


class SimulationRLAdapter:
    """
    Adapter class that connects the forex broker simulator with the RL environment.
    This provides a unified interface for training RL agents with realistic market conditions.
    """
    
    def __init__(
        self,
        broker_simulator: ForexBrokerSimulator,
        include_news_data: bool = True,
        include_sentiment: bool = True,
        include_order_book: bool = True,
        regime_transition_probability: float = 0.05,
    ):
        """
        Initialize the simulation adapter with a broker simulator instance.
        
        Args:
            broker_simulator: The forex broker simulator to use
            include_news_data: Whether to include news events in the state
            include_sentiment: Whether to include sentiment data in the state
            include_order_book: Whether to include order book data in the state
            regime_transition_probability: Probability of regime transitions
        """
        self.broker_simulator = broker_simulator
        self.include_news_data = include_news_data
        self.include_sentiment = include_sentiment
        self.include_order_book = include_order_book
        self.regime_transition_probability = regime_transition_probability
        self.current_step = 0
        self.max_steps = 10000
        self._setup_simulator()
    
    def _setup_simulator(self):
        """Configure the simulator with the appropriate settings."""
        # Initialize simulation parameters
        self.current_regime = MarketRegimeType.NORMAL
        self.pending_events = []
        self.news_impact_tracker = []
        
        # Set up realistic market conditions
        self._schedule_regime_changes()
        self._schedule_news_events()

    def _schedule_regime_changes(self):
        """Schedule market regime changes throughout the simulation period."""
        step = 0
        while step < self.max_steps:
            # Determine next regime change
            steps_to_next_change = np.random.geometric(p=self.regime_transition_probability)
            step += steps_to_next_change
            
            if step < self.max_steps:
                # Choose a new regime that's different from the current one
                regimes = list(MarketRegimeType)
                regimes.remove(self.current_regime)
                new_regime = np.random.choice(regimes)
                
                self.pending_events.append({
                    'step': step,
                    'type': 'regime_change',
                    'new_regime': new_regime,
                    'duration': int(np.random.exponential(1000))  # Average regime duration
                })

    def _schedule_news_events(self):
        """Schedule news events throughout the simulation period."""
        # Major economic releases (e.g., NFP, rate decisions)
        major_events = int(self.max_steps / 1000)  # Approximately monthly
        
        for _ in range(major_events):
            step = np.random.randint(0, self.max_steps)
            impact = np.random.choice(['high', 'very_high'])
            self.pending_events.append({
                'step': step,
                'type': 'news',
                'event_type': MarketEventType.ECONOMIC_RELEASE,
                'impact': impact,
                'duration': np.random.randint(10, 50)  # Impact lasts for 10-50 steps
            })
        
        # Regular news events (more frequent, less impact)
        minor_events = int(self.max_steps / 200)  # Several times per week
        
        for _ in range(minor_events):
            step = np.random.randint(0, self.max_steps)
            impact = np.random.choice(['low', 'medium'])
            self.pending_events.append({
                'step': step,
                'type': 'news',
                'event_type': MarketEventType.ECONOMIC_RELEASE,
                'impact': impact,
                'duration': np.random.randint(5, 20)
            })
        
        # Sort events by step
        self.pending_events.sort(key=lambda x: x['step'])

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state from the broker simulator and transform
        it into a format suitable for the RL environment.
        
        Returns:
            Dictionary containing the observations for the RL agent
        """
        # Get market data from broker simulator
        market_data = self.broker_simulator.get_market_data()
        
        # Process technical indicators
        indicators = self._calculate_technical_indicators(market_data)
        
        # Get order book data if enabled
        order_book = {}
        if self.include_order_book:
            order_book = self.broker_simulator.get_order_book_snapshot()
        
        # Get news and sentiment data if enabled
        news_data = {}
        sentiment_data = {}
        if self.include_news_data:
            news_data = self.broker_simulator.get_pending_news_events()
        if self.include_sentiment:
            sentiment_data = self.broker_simulator.get_market_sentiment()
        
        # Construct the full state
        state = {
            'price_data': market_data,
            'technical_indicators': indicators,
            'order_book': order_book,
            'news': news_data,
            'sentiment': sentiment_data,
            'regime': self.current_regime.value,
            'liquidity': self.broker_simulator.get_current_liquidity().value,
        }
        
        return state

    def _calculate_technical_indicators(self, market_data):
        """Calculate technical indicators from raw price data."""
        # Extract OHLCV data
        close_prices = market_data.get('close', [])
        if not close_prices:
            return {}
            
        # Calculate basic indicators
        indicators = {}
        
        # Moving averages (if we have enough data)
        if len(close_prices) >= 20:
            indicators['sma_5'] = np.mean(close_prices[-5:])
            indicators['sma_10'] = np.mean(close_prices[-10:])
            indicators['sma_20'] = np.mean(close_prices[-20:])
            
            # Exponential moving averages
            indicators['ema_5'] = self._calculate_ema(close_prices, 5)
            indicators['ema_10'] = self._calculate_ema(close_prices, 10)
            indicators['ema_20'] = self._calculate_ema(close_prices, 20)
            
            # RSI (14-period)
            indicators['rsi'] = self._calculate_rsi(close_prices)
            
            # Bollinger Bands (20-period, 2 standard deviations)
            middle_band = indicators['sma_20']
            std_dev = np.std(close_prices[-20:])
            indicators['bollinger_upper'] = middle_band + (2 * std_dev)
            indicators['bollinger_lower'] = middle_band - (2 * std_dev)
            
            # MACD (12, 26, 9)
            ema_12 = self._calculate_ema(close_prices, 12)
            ema_26 = self._calculate_ema(close_prices, 26)
            indicators['macd_line'] = ema_12 - ema_26
            
        return indicators
    
    def _calculate_ema(self, prices, period):
        """Calculate the Exponential Moving Average."""
        if len(prices) < period:
            return None
        
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        
        return np.convolve(weights, prices[-period*2:], mode='valid')[-1]
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate the Relative Strength Index."""
        if len(prices) < period + 1:
            return None
            
        # Calculate price changes
        deltas = np.diff(prices[-period-1:])
        
        # Calculate gains and losses
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))
        
        # Calculate average gain and loss
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100  # No losses, RSI = 100
            
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def step(self, action):
        """
        Take a step in the simulation based on the agent's action.
        
        Args:
            action: The action chosen by the RL agent
            
        Returns:
            next_state, reward, done, info
        """
        # Process any pending events that should occur at this step
        self._process_pending_events()
        
        # Execute the action in the broker simulator
        execution_result = self.broker_simulator.execute_action(action)
        
        # Calculate reward based on execution result
        reward = self._calculate_reward(execution_result)
        
        # Get the next state
        next_state = self.get_state()
        
        # Check if the episode is done
        done = self.current_step >= self.max_steps
        
        # Increment the step counter
        self.current_step += 1
        
        # Additional info for debugging and analysis
        info = {
            'execution_result': execution_result,
            'current_regime': self.current_regime.value,
            'active_news_events': self.news_impact_tracker
        }
        
        return next_state, reward, done, info

    def _process_pending_events(self):
        """Process any events that are scheduled for the current step."""
        # Check for events at the current step
        current_events = [e for e in self.pending_events if e['step'] == self.current_step]
        
        for event in current_events:
            if event['type'] == 'regime_change':
                # Update the current regime
                self.current_regime = event['new_regime']
                logger.info(f"Market regime changed to {self.current_regime.value} at step {self.current_step}")
                
                # Update the broker simulator
                self.broker_simulator.set_market_regime(self.current_regime)
                
            elif event['type'] == 'news':
                # Add to active news events
                self.news_impact_tracker.append({
                    'event_type': event['event_type'],
                    'impact': event['impact'],
                    'start_step': self.current_step,
                    'end_step': self.current_step + event['duration']
                })
                
                # Update the broker simulator
                self.broker_simulator.apply_news_event(
                    event_type=event['event_type'],
                    impact=event['impact']
                )
                
        # Remove processed events from pending list
        self.pending_events = [e for e in self.pending_events if e['step'] != self.current_step]
        
        # Update news impact tracker - remove expired events
        self.news_impact_tracker = [
            event for event in self.news_impact_tracker 
            if event['end_step'] > self.current_step
        ]

    def _calculate_reward(self, execution_result):
        """
        Calculate the reward for the agent based on the execution result.
        
        This includes PnL, risk-adjusted returns, and penalties for excessive trading or risk.
        """
        # Extract data from execution result
        pnl = execution_result.get('realized_pnl', 0)
        unrealized_pnl = execution_result.get('unrealized_pnl', 0)
        spread_cost = execution_result.get('spread_cost', 0)
        slippage = execution_result.get('slippage', 0)
        
        # Base reward is the profit/loss
        reward = pnl
        
        # Add a smaller component for unrealized PnL to encourage holding profitable positions
        reward += unrealized_pnl * 0.1
        
        # Penalize trading costs
        reward -= spread_cost
        reward -= slippage
        
        # Risk-adjustment factors
        position_size = execution_result.get('position_size', 0)
        max_drawdown = execution_result.get('max_drawdown', 0)
        
        # Penalize excessive risk
        if max_drawdown > 0.05:  # More than 5% drawdown
            risk_penalty = max_drawdown * 10
            reward -= risk_penalty
        
        # Penalize excessive position sizes
        if position_size > 0.5:  # Using more than 50% of available capital
            size_penalty = (position_size - 0.5) * 5
            reward -= size_penalty
            
        return reward

    def reset(self):
        """Reset the simulation to its initial state."""
        self.current_step = 0
        self._setup_simulator()
        self.broker_simulator.reset()
        
        # Return initial state
        return self.get_state()


class EnhancedForexTradingEnv(ForexTradingEnvironment):
    """
    Enhanced Forex Trading Environment that integrates with the simulation adapter
    to provide a more realistic training environment for RL agents.
    """
    
    def __init__(
        self,
        simulation_adapter: SimulationRLAdapter,
        observation_space_config: dict = None,
        action_space_config: dict = None,
        reward_config: dict = None,
        **kwargs
    ):
        """
        Initialize the enhanced forex trading environment.
        
        Args:
            simulation_adapter: Adapter connecting to the forex broker simulator
            observation_space_config: Configuration for observation space
            action_space_config: Configuration for action space
            reward_config: Configuration for reward calculation
        """
        self.simulation_adapter = simulation_adapter
        
        # Call parent constructor with additional configs
        super().__init__(
            observation_space_config=observation_space_config,
            action_space_config=action_space_config,
            reward_config=reward_config,
            **kwargs
        )
        
    def reset(self):
        """Reset the environment."""
        # Reset the simulation adapter
        initial_state = self.simulation_adapter.reset()
        
        # Process the state to match the gym observation space
        observation = self._process_state_for_observation(initial_state)
        
        return observation
        
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: The action from the RL agent
            
        Returns:
            observation, reward, done, info
        """
        # Translate the gym action to broker simulator action format
        broker_action = self._translate_action(action)
        
        # Step the simulation adapter
        next_state, reward, done, info = self.simulation_adapter.step(broker_action)
        
        # Process the state to match the gym observation space
        observation = self._process_state_for_observation(next_state)
        
        return observation, reward, done, info
    
    def _translate_action(self, action):
        """
        Translate the agent's action from gym format to broker simulator format.
        
        Args:
            action: The action from the gym action space
            
        Returns:
            Action in the format expected by the broker simulator
        """
        # Implement action translation logic based on your action space
        # This is just an example and should be adapted to your specific action space
        if isinstance(action, int):
            # Discrete action space
            action_map = {
                0: {'action_type': 'buy', 'position_size': 0.1},
                1: {'action_type': 'sell', 'position_size': 0.1},
                2: {'action_type': 'hold', 'position_size': 0},
                3: {'action_type': 'close', 'position_size': 0},
            }
            return action_map.get(action, {'action_type': 'hold', 'position_size': 0})
        
        elif isinstance(action, np.ndarray):
            # Continuous action space
            # Example: [action_type_logits, position_size]
            action_type_idx = np.argmax(action[0:4])
            position_size = max(0, min(1, action[4]))  # Clip between 0 and 1
            
            action_types = ['buy', 'sell', 'hold', 'close']
            return {
                'action_type': action_types[action_type_idx],
                'position_size': position_size
            }
        
        return {'action_type': 'hold', 'position_size': 0}
    
    def _process_state_for_observation(self, state):
        """
        Process the state from the simulation adapter into the format
        expected by the gym observation space.
        
        Args:
            state: Raw state from the simulation adapter
            
        Returns:
            Observation that conforms to the gym observation space
        """
        # This implementation depends on your specific observation space
        # Here's an example for a flat vector observation space:
        
        # Extract relevant features from the state
        price_data = state.get('price_data', {})
        technical_indicators = state.get('technical_indicators', {})
        regime = state.get('regime')
        
        # Convert categorical data to one-hot encoding
        regime_encoding = self._one_hot_encode_regime(regime)
        
        # Combine numerical features
        numerical_features = []
        
        # Add price data
        if price_data:
            for key in ['open', 'high', 'low', 'close', 'volume']:
                if key in price_data:
                    numerical_features.append(price_data[key][-1])  # Latest value
        
        # Add technical indicators
        if technical_indicators:
            for key, value in technical_indicators.items():
                if value is not None:
                    numerical_features.append(value)
        
        # Combine all features
        all_features = np.concatenate([
            np.array(numerical_features, dtype=np.float32),
            np.array(regime_encoding, dtype=np.float32)
        ])
        
        return all_features
    
    def _one_hot_encode_regime(self, regime):
        """Convert market regime to one-hot encoding."""
        regimes = ['trending', 'ranging', 'volatile', 'breakout', 'crisis', 'normal']
        encoding = [0] * len(regimes)
        
        if regime in regimes:
            encoding[regimes.index(regime)] = 1
            
        return encoding
