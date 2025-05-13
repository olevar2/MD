"""
Forex Trading Environment

This module provides a reinforcement learning environment for forex trading.
"""

import numpy as np
import pandas as pd
from gym import spaces
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from datetime import datetime, timedelta
import random

from core_foundations.utils.logger import get_logger
from common_lib.simulation.interfaces import (
    IBrokerSimulator, IMarketRegimeSimulator, MarketRegimeType
)
from common_lib.reinforcement.interfaces import IRLEnvironment

from .base_environment import BaseRLEnvironment
from .reward.base_reward import CompositeReward
from .reward.risk_adjusted_reward import create_risk_adjusted_reward
from .reward.pnl_reward import create_pnl_reward
from .reward.custom_reward import create_custom_reward, NewsAlignmentReward
from .state.observation_space import ObservationSpaceBuilder
from .state.state_representation import create_forex_state_representation

logger = get_logger(__name__)


class ForexTradingEnvironment(BaseRLEnvironment):
    """
    Reinforcement learning environment for forex trading.
    
    This environment simulates forex trading with support for:
    - Multi-timeframe observation space
    - Comprehensive state representation with technical indicators
    - Risk-adjusted reward function
    - Integration with broker simulator for realistic market conditions
    - Support for curriculum learning
    """
    
    def __init__(
        self,
        broker_simulator: IBrokerSimulator,
        symbol: str = "EUR/USD",
        timeframes: List[str] = None,  # e.g. ["1m", "5m", "15m", "1h"]
        lookback_periods: int = 50,
        features: List[str] = None,
        position_sizing_type: str = "fixed",  # "fixed", "dynamic", "risk_based"
        max_position_size: float = 1.0,  # in lots
        trading_fee_percent: float = 0.002,
        reward_mode: str = "risk_adjusted",  # "pnl", "risk_adjusted", "custom"
        risk_free_rate: float = 0.02,  # Annual risk-free rate for Sharpe ratio
        episode_timesteps: int = 1000,
        time_step_seconds: int = 60,  # 1 minute by default
        random_episode_start: bool = True,
        curriculum_level: int = 0,  # 0 = easiest, higher = harder
        include_broker_state: bool = True,
        include_order_book: bool = True,
        include_technical_indicators: bool = True,
        include_news_sentiment: bool = True,  # Include news and sentiment features
        news_sentiment_simulator: Optional[Any] = None,  # News simulator
        custom_reward_function: Optional[Callable] = None,
        observation_normalization: bool = True,
        custom_indicators: Dict[str, Callable] = None
    ):
        """
        Initialize the forex trading environment.
        
        Args:
            broker_simulator: Forex broker simulator instance
            symbol: Trading symbol
            timeframes: List of timeframes to include in observation
            lookback_periods: Number of past periods to include in observation
            features: Features to include from market data
            position_sizing_type: How to determine position sizes
            max_position_size: Maximum position size in lots
            trading_fee_percent: Trading fee as percentage
            reward_mode: Type of reward calculation
            risk_free_rate: Annual risk-free rate for risk-adjusted metrics
            episode_timesteps: Maximum timesteps per episode
            time_step_seconds: Seconds to advance per step
            random_episode_start: Whether to start episodes at random positions
            curriculum_level: Difficulty level for curriculum learning
            include_broker_state: Include broker state in observation
            include_order_book: Include order book data in observation
            include_technical_indicators: Include technical indicators in observation
            include_news_sentiment: Include news and sentiment features
            news_sentiment_simulator: Optional news sentiment simulator instance
            custom_reward_function: Optional custom reward function
            observation_normalization: Whether to normalize observations
            custom_indicators: Optional dictionary of custom indicator functions
        """
        super().__init__(
            episode_timesteps=episode_timesteps,
            random_episode_start=random_episode_start,
            observation_normalization=observation_normalization
        )
        
        self.broker_simulator = broker_simulator
        self.symbol = symbol
        self.timeframes = timeframes if timeframes else ["1m"]
        self.lookback_periods = lookback_periods
        self.features = features if features else ["open", "high", "low", "close", "volume"]
        self.position_sizing_type = position_sizing_type
        self.max_position_size = max_position_size
        self.trading_fee_percent = trading_fee_percent
        self.reward_mode = reward_mode
        self.risk_free_rate = risk_free_rate
        self.time_step_seconds = time_step_seconds
        self.curriculum_level = curriculum_level
        self.include_broker_state = include_broker_state
        self.include_order_book = include_order_book
        self.include_technical_indicators = include_technical_indicators
        self.include_news_sentiment = include_news_sentiment
        self.news_sentiment_simulator = news_sentiment_simulator
        self.custom_reward_function = custom_reward_function
        self.custom_indicators = custom_indicators or {}
        
        # Internal state
        self.account_balance = 10000.0  # Starting balance
        self.current_pnl = 0.0
        self.current_position = 0.0
        self.trade_history = []
        self.episode_returns = []
        self.total_trades = 0
        self.profitable_trades = 0
        self.last_transaction_cost = 0.0
        self.market_data = {}
        
        # Set up observation and action spaces
        self._setup_spaces()
        
        # Set up reward function
        self._setup_reward_function()
        
        # Set up state representation
        self._setup_state_representation()
    
    def _setup_spaces(self):
        """Set up the observation and action spaces."""
        # Use the ObservationSpaceBuilder to construct the observation space
        builder = ObservationSpaceBuilder()
        
        # Add market data features
        builder.add_market_data_features(
            timeframes=self.timeframes,
            features=self.features,
            lookback_periods=self.lookback_periods
        )
        
        # Add technical indicators if enabled
        if self.include_technical_indicators:
            indicators = ['sma5', 'sma20', 'ema5', 'ema20', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_lower']
            builder.add_technical_indicators(
                timeframes=self.timeframes,
                indicators=indicators
            )
            
            # Add custom indicators
            if self.custom_indicators:
                builder.add_custom_features(
                    name='custom_indicators',
                    num_features=len(self.custom_indicators) * len(self.timeframes)
                )
        
        # Add order book features if enabled
        if self.include_order_book:
            builder.add_order_book(levels=5)
        
        # Add broker state features if enabled
        if self.include_broker_state:
            builder.add_broker_state()
        
        # Add news sentiment features if enabled
        if self.include_news_sentiment and self.news_sentiment_simulator:
            news_sentiment_dim = self._get_news_sentiment_dimension()
            builder.add_news_sentiment(news_sentiment_dim)
        
        # Build the observation space
        self.observation_space, self.feature_dimensions = builder.build()
        
        # Define action space
        # Actions: [position_size_pct] where -1.0 = max short, 0 = no position, 1.0 = max long
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
    
    def _setup_reward_function(self):
        """Set up the reward function based on the selected mode."""
        if self.custom_reward_function:
            self.reward_function = create_custom_reward(self.custom_reward_function)
        elif self.reward_mode == "risk_adjusted":
            self.reward_function = create_risk_adjusted_reward(
                risk_free_rate=self.risk_free_rate,
                curriculum_level=self.curriculum_level
            )
            
            # Add news alignment component if using news
            if self.include_news_sentiment and self.news_sentiment_simulator:
                self.reward_function.add_component(NewsAlignmentReward(weight=0.3))
        else:  # Default to PnL reward
            self.reward_function = create_pnl_reward(trading_fee_weight=1.0)
    
    def _setup_state_representation(self):
        """Set up the state representation."""
        news_sentiment_dim = self._get_news_sentiment_dimension() if self.include_news_sentiment else 0
        
        self.state_representation, _ = create_forex_state_representation(
            timeframes=self.timeframes,
            features=self.features,
            lookback_periods=self.lookback_periods,
            include_technical_indicators=self.include_technical_indicators,
            include_broker_state=self.include_broker_state,
            include_order_book=self.include_order_book,
            include_news_sentiment=self.include_news_sentiment,
            news_sentiment_dimension=news_sentiment_dim
        )
    
    def _get_news_sentiment_dimension(self) -> int:
        """Get the dimension size for news and sentiment features."""
        # Basic features: recent events count, average sentiment, highest impact
        base_dim = 3
        
        # Impact levels (assuming 3 levels: LOW, MEDIUM, HIGH)
        impact_levels = 3
        
        # Sentiment levels (assuming 3 levels: NEGATIVE, NEUTRAL, POSITIVE)
        sentiment_levels = 3
        
        # Currency specific sentiment
        currencies = set()
        if "/" in self.symbol:
            currencies.update(self.symbol.split('/'))
        
        # Total features
        total_dim = base_dim + impact_levels + sentiment_levels + len(currencies)
        return total_dim
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation based on the environment state."""
        # Prepare data for state representation
        data = {}
        
        # Market data for each timeframe
        market_data = {}
        for tf in self.timeframes:
            market_data[tf] = self.broker_simulator.get_historical_data(
                symbol=self.symbol,
                timeframe=tf,
                periods=self.lookback_periods + 50,  # Extra periods for indicator calculation
                end_time=self.current_timestamp
            )
        data['market_data'] = market_data
        
        # Broker state
        if self.include_broker_state:
            data['broker_state'] = {
                'account_balance': self.account_balance,
                'current_position': self.current_position,
                'current_pnl': self.current_pnl,
                'total_trades': self.total_trades,
                'profitable_trades': self.profitable_trades,
                'max_position_size': self.max_position_size
            }
        
        # Order book
        if self.include_order_book:
            data['order_book'] = self.broker_simulator.get_order_book(
                symbol=self.symbol,
                timestamp=self.current_timestamp
            )
        
        # News sentiment
        if self.include_news_sentiment and self.news_sentiment_simulator:
            data['news_sentiment'] = {
                'events': self.news_sentiment_simulator.get_recent_events(
                    self.current_timestamp,
                    lookback_hours=24,
                    relevant_currencies=self.symbol.replace('/', ',')
                )
            }
        
        # Get state representation
        return self.state_representation.get_state(data)
    
    def _take_action(self, action: np.ndarray) -> Tuple[float, Dict]:
        """
        Execute the action in the trading environment.
        
        Args:
            action: Normalized position size (-1.0 to 1.0)
            
        Returns:
            reward: The reward for this action
            info: Additional information
        """
        # Convert normalized action to actual position size
        target_position_size = float(action[0]) * self.max_position_size
        
        # Determine size change needed
        position_change = target_position_size - self.current_position
        
        # Execute trade if there's a position change
        trade_result = None
        if abs(position_change) > 0.01:  # Minimum position change threshold
            # Determine order details
            side = "buy" if position_change > 0 else "sell"
            size = abs(position_change)
            
            # Execute order through broker simulator
            trade_result = self.broker_simulator.execute_order(
                symbol=self.symbol,
                side=side,
                order_type="market",
                size=size,
                timestamp=self.current_timestamp
            )
            
            # Update trade statistics
            self.total_trades += 1
            
            # Record the trade
            self.trade_history.append({
                'timestamp': self.current_timestamp,
                'side': side,
                'size': size,
                'price': trade_result['avg_price'],
                'fee': trade_result['fee']
            })
            
            # Store transaction cost for reward calculation
            self.last_transaction_cost = trade_result['fee']
        else:
            self.last_transaction_cost = 0.0
        
        # Update current position
        self.current_position = target_position_size
        
        # Update P&L based on position and price changes
        current_price = self.broker_simulator.get_current_price(self.symbol, self.current_timestamp)
        
        if trade_result:
            # Calculate P&L from the trade
            trade_pnl = trade_result['realized_pnl']
            
            # Apply trading fees
            fee = trade_result['fee']
            
            # Update account balance
            self.account_balance += trade_pnl - fee
            
            # Update current P&L with this step's change
            self.current_pnl = trade_pnl - fee
            
            # Track profitability
            if trade_pnl > 0:
                self.profitable_trades += 1
        else:
            # No trade executed, P&L comes from holding position
            if self.current_position != 0:
                # Calculate P&L from price change on the existing position
                price_change = self.broker_simulator.get_price_change(
                    symbol=self.symbol,
                    from_time=self.current_timestamp - timedelta(seconds=self.time_step_seconds),
                    to_time=self.current_timestamp
                )
                
                holding_pnl = self.current_position * price_change
                self.account_balance += holding_pnl
                self.current_pnl = holding_pnl
            else:
                self.current_pnl = 0.0
        
        # Keep track of returns for risk metrics
        self.episode_returns.append(self.current_pnl)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Return reward and info
        info = {
            'account_balance': self.account_balance,
            'current_position': self.current_position,
            'current_pnl': self.current_pnl,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'win_rate': self.profitable_trades / max(1, self.total_trades),
            'current_price': current_price,
        }
        
        return reward, info
    
    def _calculate_reward(self) -> float:
        """Calculate the reward based on the current state and action."""
        return self.reward_function.calculate(self)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment by executing the given action.
        
        Args:
            action: The action to take
            
        Returns:
            observation: The new observation
            reward: The reward for the action
            done: Whether the episode is finished
            info: Additional information
        """
        # Advance the simulation time
        self.current_timestamp += timedelta(seconds=self.time_step_seconds)
        
        # Call the parent step method
        observation, reward, done, info = super().step(action)
        
        # Check for additional termination conditions
        if self.account_balance <= self.account_balance * 0.1:  # 90% drawdown
            done = True
            # Apply penalty for bankruptcy
            reward -= 10.0
        
        return observation, reward, done, info
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to start a new episode.
        
        Returns:
            observation: The initial observation
        """
        # Reset environment state
        self.account_balance = 10000.0
        self.current_pnl = 0.0
        self.current_position = 0.0
        self.trade_history = []
        self.episode_returns = []
        self.total_trades = 0
        self.profitable_trades = 0
        self.last_transaction_cost = 0.0
        
        # Reset market data cache
        self.market_data = {}
        
        # Set initial timestamp
        if self.random_episode_start:
            # Start at a random time within the available data range
            available_times = self.broker_simulator.get_available_times(self.symbol)
            if available_times:
                # Choose a random time that leaves enough room for an episode
                max_idx = max(0, len(available_times) - self.max_episode_steps - 1)
                if max_idx > 0:
                    start_idx = random.randint(0, max_idx)
                    self.current_timestamp = available_times[start_idx]
                else:
                    self.current_timestamp = available_times[0]
            else:
                # Default to current time if no data available
                self.current_timestamp = datetime.now()
        else:
            # Start at the beginning of available data
            available_times = self.broker_simulator.get_available_times(self.symbol)
            if available_times:
                self.current_timestamp = available_times[0]
            else:
                self.current_timestamp = datetime.now()
        
        # Set the simulation in the broker to the correct timestamp
        self.broker_simulator.set_current_time(self.current_timestamp)
        
        # Also set the news simulator to the same time if available
        if self.news_sentiment_simulator:
            self.news_sentiment_simulator.set_current_time(self.current_timestamp)
        
        # Configure environment difficulty based on curriculum level
        self._configure_difficulty()
        
        # Call the parent reset method
        return super().reset()
    
    def _configure_difficulty(self):
        """Configure the environment difficulty based on the curriculum level."""
        if self.curriculum_level == 0:
            # Easiest: Stable market, low volatility, no extreme events
            self.broker_simulator.set_market_regime(MarketRegimeType.NORMAL)
            self.broker_simulator.set_volatility_factor(0.5)
            self.broker_simulator.enable_extreme_events(False)
        
        elif self.curriculum_level == 1:
            # Medium: Mixed regimes, normal volatility
            self.broker_simulator.set_market_regime(MarketRegimeType.RANGING)
            self.broker_simulator.set_volatility_factor(1.0)
            self.broker_simulator.enable_extreme_events(False)
        
        elif self.curriculum_level == 2:
            # Hard: More challenging regimes, higher volatility
            regimes = [MarketRegimeType.TRENDING, MarketRegimeType.VOLATILE, MarketRegimeType.RANGING]
            self.broker_simulator.set_market_regime(random.choice(regimes))
            self.broker_simulator.set_volatility_factor(1.5)
            self.broker_simulator.enable_extreme_events(True)
            self.broker_simulator.set_extreme_event_probability(0.01)  # 1% chance per step
        
        else:  # Level 3+
            # Very hard: All regimes including crisis, high volatility, frequent extreme events
            regimes = list(MarketRegimeType)
            self.broker_simulator.set_market_regime(random.choice(regimes))
            self.broker_simulator.set_volatility_factor(2.0)
            self.broker_simulator.enable_extreme_events(True)
            self.broker_simulator.set_extreme_event_probability(0.02)  # 2% chance per step
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode != 'human':
            return
        
        print(f"Step: {self.current_step}, Time: {self.current_timestamp}")
        print(f"Account Balance: ${self.account_balance:.2f}")
        print(f"Current Position: {self.current_position:.4f} lots")
        print(f"Current P&L: ${self.current_pnl:.2f}")
        print(f"Total Trades: {self.total_trades}, Profitable: {self.profitable_trades}")
        if self.total_trades > 0:
            print(f"Win Rate: {self.profitable_trades / self.total_trades:.2%}")
        print(f"Current Price: {self.broker_simulator.get_current_price(self.symbol, self.current_timestamp)}")
        print(f"Current Market Regime: {self.broker_simulator.get_current_market_regime().name}")
        print("-" * 50)
    
    def get_episode_summary(self) -> Dict:
        """
        Get summary statistics for the completed episode.
        
        Returns:
            Dictionary with episode summary statistics
        """
        if not self.trade_history:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_return_per_trade": 0.0,
                "reward_components": {},
                "final_balance": self.account_balance
            }
        
        # Calculate performance metrics
        from .reward.risk_adjusted_reward import calculate_sharpe_ratio, calculate_max_drawdown
        
        total_return = sum(self.episode_returns)
        sharpe = calculate_sharpe_ratio(self.episode_returns, self.risk_free_rate) if len(self.episode_returns) > 1 else 0.0
        max_dd = calculate_max_drawdown(self.episode_returns)
        win_rate = self.profitable_trades / max(1, self.total_trades)
        avg_return = total_return / max(1, self.total_trades)
        
        # Get reward component contributions
        reward_components = {}
        if hasattr(self.reward_function, 'get_component_values'):
            component_values = self.reward_function.get_component_values(self)
            for name, value in component_values.items():
                reward_components[name] = value
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "total_trades": self.total_trades,
            "win_rate": win_rate,
            "avg_return_per_trade": avg_return,
            "reward_components": reward_components,
            "final_balance": self.account_balance
        }