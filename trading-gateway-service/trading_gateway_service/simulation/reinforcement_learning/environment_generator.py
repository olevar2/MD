"""
Environment Generator for Reinforcement Learning Trading Agents

This module provides comprehensive environment generation capabilities for training
reinforcement learning agents on forex trading tasks. It supports creating diverse
trading scenarios with configurable parameters and customizable reward functions.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import random
from enum import Enum
import gym
from gym import spaces
import logging
from dataclasses import dataclass

from trading_gateway_service.simulation.advanced_market_regime_simulator import (
    AdvancedMarketRegimeSimulator, MarketCondition, SimulationScenario
)
from trading_gateway_service.simulation.forex_broker_simulator import (
    ForexBrokerSimulator, OrderType, OrderSide, Order
)
from trading_gateway_service.simulation.news_sentiment_simulator import (
    NewsAndSentimentSimulator
)
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class ActionType(str, Enum):
    """Types of actions available to the trading agent."""
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    HOLD = "hold"


class RewardType(str, Enum):
    """Types of reward functions available."""
    PNL_CHANGE = "pnl_change"  # Reward based on change in P&L
    RISK_ADJUSTED = "risk_adjusted"  # Sharpe-ratio style reward
    DRAWDOWN_PENALIZED = "drawdown_penalized"  # PnL with drawdown penalty
    DIRECTIONAL_CORRECTNESS = "directional_correctness"  # Reward for correct direction
    CUSTOM = "custom"  # Custom reward function


@dataclass
class EnvConfiguration:
    """Configuration for the trading environment."""
    symbols: List[str]
    timeframe: str = "1h"
    initial_balance: float = 10000.0
    leverage: float = 100.0
    commission_rate: float = 0.0001  # 0.01%
    action_type: str = "discrete"  # 'discrete' or 'continuous'
    reward_function: RewardType = RewardType.PNL_CHANGE
    max_episode_steps: int = 1000
    normalize_observations: bool = True
    include_technical_indicators: bool = True
    include_order_book_features: bool = False
    include_market_regime_features: bool = True
    include_time_features: bool = True
    window_size: int = 20  # Number of past observations included
    random_episode_start: bool = True
    transaction_cost_pct: float = 0.0001


class ForexTradingEnvironment(gym.Env):
    """
    Reinforcement learning environment for forex trading using the advanced
    market regime simulator and forex broker simulator.
    
    Features:
    - Support for multiple symbols
    - Realistic market simulation
    - Customizable observation space including technical indicators
    - Various reward functions
    - Detailed trade tracking and analytics
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        market_simulator: AdvancedMarketRegimeSimulator,
        broker_simulator: ForexBrokerSimulator,
        config: EnvConfiguration,
        custom_reward_function: Optional[Callable] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the forex trading environment.
        
        Args:
            market_simulator: Advanced market regime simulator
            broker_simulator: Forex broker simulator
            config: Environment configuration
            custom_reward_function: Custom reward function (optional)
            random_seed: Random seed for reproducibility
        """
        super(ForexTradingEnvironment, self).__init__()
        
        self.market_simulator = market_simulator
        self.broker_simulator = broker_simulator
        self.config = config
        self.custom_reward_function = custom_reward_function
        
        # Set random seed
        if random_seed is not None:
            self.seed(random_seed)
            
        # Initialize state variables
        self.current_step = 0
        self.current_time = datetime.now()
        self.historical_data = {}
        self.account_history = []
        self.trades = []
        self.done = False
        
        # Set up action space
        self._setup_action_space()
        
        # Set up observation space
        self._setup_observation_space()
    
    def _setup_action_space(self):
        """Set up the action space based on configuration."""
        if self.config.action_type == "discrete":
            # Discrete actions: Buy, Sell, Close, Hold for each symbol
            self.action_space = spaces.Discrete(len(ActionType) * len(self.config.symbols))
        else:  # Continuous
            # For each symbol: position size (-1 to 1) where:
            # -1 = max short position, 0 = no position, 1 = max long position
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(len(self.config.symbols),),
                dtype=np.float32
            )
            
    def _setup_observation_space(self):
        """Set up the observation space based on configuration."""
        # Calculate the number of features per symbol
        num_features = 6  # OHLCV + spread
        
        # Add technical indicators if enabled
        if self.config.include_technical_indicators:
            num_features += 10  # Common indicators like RSI, MACD, etc.
            
        # Add order book features if enabled
        if self.config.include_order_book_features:
            num_features += 10  # Top levels, depth, etc.
            
        # Add market regime features if enabled
        if self.config.include_market_regime_features:
            num_features += 5  # Regime type, volatility, etc.
            
        # Add time features if enabled
        if self.config.include_time_features:
            num_features += 5  # Hour of day, day of week, etc.
            
        # Add position features
        num_features += 3  # Position size, P&L, entry price
        
        # Create observation space with history window
        total_features = num_features * len(self.config.symbols)
        self.observation_shape = (self.config.window_size, total_features)
        
        # Observation space is a continuous box normalized between -1 and 1
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0, 
            shape=self.observation_shape,
            dtype=np.float32
        )
        
    def seed(self, seed=None):
        """Set the random seed."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return [seed]
    
    def reset(self):
        """
        Reset the environment to start a new episode.
        
        Returns:
            Initial observation
        """
        # Reset broker simulator to initial state
        self.broker_simulator = ForexBrokerSimulator(
            balance=self.config.initial_balance,
            leverage=self.config.leverage,
            commission_per_lot=self.config.commission_rate * 100000  # Convert to per lot
        )
        
        # Generate market data for this episode
        self._generate_episode_data()
        
        # Reset state variables
        self.current_step = 0
        self.account_history = []
        self.trades = []
        self.done = False
        
        # Save initial account state
        self._record_account_state()
        
        # Get initial observation
        return self._get_observation()
    
    def _generate_episode_data(self):
        """Generate market data for the current episode."""
        self.historical_data = {}
        
        # Set episode start time
        if self.config.random_episode_start:
            # Random start time in the last year
            start_time = datetime.now() - timedelta(days=random.randint(1, 365))
        else:
            start_time = datetime.now() - timedelta(days=30)  # Default to 30 days ago
        
        self.current_time = start_time
        
        # Calculate end time based on max steps and timeframe
        end_time = self._calculate_end_time()
        
        # Generate data for each symbol
        for symbol in self.config.symbols:
            # Generate market data using the simulator
            data = self.market_simulator.generate_market_data(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                timeframe=self.config.timeframe,
                include_indicators=self.config.include_technical_indicators
            )
            
            self.historical_data[symbol] = data
    
    def _calculate_end_time(self) -> datetime:
        """Calculate the end time for data generation based on config."""
        # Parse timeframe
        if self.config.timeframe.endswith('m'):
            minutes = int(self.config.timeframe[:-1])
            end_time = self.current_time + timedelta(minutes=minutes * self.config.max_episode_steps)
        elif self.config.timeframe.endswith('h'):
            hours = int(self.config.timeframe[:-1])
            end_time = self.current_time + timedelta(hours=hours * self.config.max_episode_steps)
        elif self.config.timeframe.endswith('d'):
            days = int(self.config.timeframe[:-1])
            end_time = self.current_time + timedelta(days=days * self.config.max_episode_steps)
        else:
            # Default to 1 hour if timeframe format is unknown
            end_time = self.current_time + timedelta(hours=self.config.max_episode_steps)
            
        return end_time
    
    def _advance_time(self):
        """Advance the simulation time based on the configured timeframe."""
        # Parse timeframe
        if self.config.timeframe.endswith('m'):
            minutes = int(self.config.timeframe[:-1])
            self.current_time += timedelta(minutes=minutes)
        elif self.config.timeframe.endswith('h'):
            hours = int(self.config.timeframe[:-1])
            self.current_time += timedelta(hours=hours)
        elif self.config.timeframe.endswith('d'):
            days = int(self.config.timeframe[:-1])
            self.current_time += timedelta(days=days)
        else:
            # Default to 1 hour if timeframe format is unknown
            self.current_time += timedelta(hours=1)
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Agent's action
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Process the action
        self._process_action(action)
        
        # Advance the simulation time
        self._advance_time()
        self.current_step += 1
        
        # Update broker simulator with new prices
        self._update_prices()
        
        # Record the new account state
        self._record_account_state()
        
        # Check if episode is done
        self._check_episode_done()
        
        # Get the current observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Compile info dict
        info = self._get_step_info()
        
        return observation, reward, self.done, info
    
    def _process_action(self, action):
        """Process the agent's action."""
        if self.config.action_type == "discrete":
            self._process_discrete_action(action)
        else:
            self._process_continuous_action(action)
    
    def _process_discrete_action(self, action_idx):
        """
        Process a discrete action.
        
        Args:
            action_idx: Index of the action
        """
        # Calculate which symbol and action type this corresponds to
        num_action_types = len(ActionType)
        symbol_idx = action_idx // num_action_types
        action_type_idx = action_idx % num_action_types
        
        if symbol_idx >= len(self.config.symbols):
            logger.warning(f"Invalid symbol index: {symbol_idx}")
            return
        
        symbol = self.config.symbols[symbol_idx]
        action_type = list(ActionType)[action_type_idx]
        
        # Execute the action
        if action_type == ActionType.BUY:
            self._open_position(symbol, OrderSide.BUY)
        elif action_type == ActionType.SELL:
            self._open_position(symbol, OrderSide.SELL)
        elif action_type == ActionType.CLOSE:
            self._close_position(symbol)
        # HOLD requires no action
    
    def _process_continuous_action(self, actions):
        """
        Process continuous actions.
        
        Args:
            actions: Array of position sizes for each symbol (-1 to 1)
        """
        for i, symbol in enumerate(self.config.symbols):
            if i >= len(actions):
                continue
                
            target_position = actions[i]
            
            # Get current position
            positions = self.broker_simulator.get_positions()
            current_position = 0.0
            
            if symbol in positions:
                current_position = positions[symbol]["quantity"]
            
            # Calculate the difference to determine action
            position_diff = target_position - current_position
            
            # Skip if the change is very small
            if abs(position_diff) < 0.1:
                continue
                
            # Close existing position if direction changed
            if (current_position > 0 and target_position < 0) or \
               (current_position < 0 and target_position > 0):
                self._close_position(symbol)
                
                # Open new position in the opposite direction
                side = OrderSide.BUY if target_position > 0 else OrderSide.SELL
                self._open_position(symbol, side, abs(target_position))
            
            # Adjust existing position
            elif abs(position_diff) > 0:
                if position_diff > 0:
                    # Increase position or open new long
                    self._open_position(symbol, OrderSide.BUY, position_diff)
                else:
                    # Increase short position or open new short
                    self._open_position(symbol, OrderSide.SELL, abs(position_diff))
    
    def _open_position(self, symbol, side, size_scale=1.0):
        """
        Open a trading position.
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            size_scale: Scale factor for position size (0-1)
        """
        # Get current price
        current_price = self._get_current_price(symbol)
        if current_price is None:
            return
            
        # Calculate position size (in standard lots)
        account_summary = self.broker_simulator.get_account_summary()
        equity = account_summary["equity"]
        
        # Risk 2% of equity per trade by default, scaled by size_scale
        risk_amount = equity * 0.02 * size_scale
        position_size = risk_amount * self.config.leverage / current_price
        
        # Create and place order
        order = Order(
            symbol=symbol,
            order_type=OrderType.MARKET,
            side=side,
            quantity=position_size
        )
        
        try:
            order_id = self.broker_simulator.place_order(order)
            
            # Record the trade
            self.trades.append({
                "order_id": order_id,
                "symbol": symbol,
                "side": side.value,
                "quantity": position_size,
                "price": current_price,
                "timestamp": self.current_time,
                "step": self.current_step
            })
        except Exception as e:
            logger.error(f"Error placing order: {e}")
    
    def _close_position(self, symbol):
        """
        Close an open position.
        
        Args:
            symbol: Trading symbol
        """
        try:
            positions = self.broker_simulator.get_positions()
            if symbol in positions:
                # Close the position
                self.broker_simulator.close_position(symbol)
                
                # Record the trade
                self.trades.append({
                    "order_id": None,
                    "symbol": symbol,
                    "side": "close",
                    "quantity": 0,  # Position closed
                    "price": self._get_current_price(symbol),
                    "timestamp": self.current_time,
                    "step": self.current_step
                })
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def _get_current_price(self, symbol):
        """Get the current price for a symbol."""
        try:
            prices = self.broker_simulator.get_current_prices()
            if symbol in prices:
                return prices[symbol]["bid"]
            return None
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None
    
    def _update_prices(self):
        """Update broker simulator with prices at the current timestamp."""
        for symbol in self.config.symbols:
            if symbol not in self.historical_data:
                continue
                
            # Get data at current timestamp
            df = self.historical_data[symbol]
            
            # Find the closest timestamp
            closest_idx = None
            
            try:
                # Try to find exact match first
                closest_idx = df.index.get_indexer([self.current_time], method='pad')[0]
            except:
                # If not found, use the last available data point
                if len(df) > 0:
                    closest_idx = len(df) - 1
            
            if closest_idx is None or closest_idx < 0:
                continue
                
            # Get the row of data
            row = df.iloc[closest_idx]
            
            # Update the broker simulator with this price
            if hasattr(self.broker_simulator, 'set_current_price'):
                self.broker_simulator.set_current_price(
                    symbol,
                    row['close'],
                    row.get('spread', 0.0001)  # Default spread if not available
                )
    
    def _record_account_state(self):
        """Record the current account state."""
        account_summary = self.broker_simulator.get_account_summary()
        positions = self.broker_simulator.get_positions()
        
        # Add to account history
        self.account_history.append({
            "step": self.current_step,
            "timestamp": self.current_time,
            "balance": account_summary["balance"],
            "equity": account_summary["equity"],
            "margin_used": account_summary["margin_used"],
            "open_positions": len(positions),
            "unrealized_pnl": account_summary["unrealized_pnl"],
            "realized_pnl": account_summary.get("realized_pnl", 0.0)
        })
    
    def _check_episode_done(self):
        """Check if the episode is done."""
        # Check if max steps reached
        if self.current_step >= self.config.max_episode_steps:
            self.done = True
            return
            
        # Check if account is blown up (equity below minimum)
        account_summary = self.broker_simulator.get_account_summary()
        if account_summary["equity"] <= self.config.initial_balance * 0.1:  # 90% loss
            self.done = True
            return
    
    def _get_observation(self):
        """Get the current observation."""
        # Prepare a list to hold all features
        all_features = []
        
        # Process each symbol
        for symbol in self.config.symbols:
            if symbol not in self.historical_data:
                # If data not available, use zeros
                symbol_features = np.zeros(self.observation_shape[1] // len(self.config.symbols))
                all_features.append(symbol_features)
                continue
            
            # Get data for this symbol
            df = self.historical_data[symbol]
            
            # Find index range for window
            end_idx = None
            try:
                end_idx = df.index.get_indexer([self.current_time], method='pad')[0]
            except:
                if len(df) > 0:
                    end_idx = len(df) - 1
            
            if end_idx is None or end_idx < 0:
                # If no data available, use zeros
                symbol_features = np.zeros(self.observation_shape[1] // len(self.config.symbols))
                all_features.append(symbol_features)
                continue
            
            # Calculate window start
            start_idx = max(0, end_idx - self.config.window_size + 1)
            
            # Get window data
            window_data = df.iloc[start_idx:end_idx+1]
            
            # Extract features
            features = self._extract_features(window_data, symbol)
            all_features.append(features)
        
        # Combine features from all symbols
        combined_features = np.concatenate(all_features, axis=1)
        
        # Normalize if required
        if self.config.normalize_observations:
            combined_features = self._normalize_features(combined_features)
        
        # Ensure correct shape
        if combined_features.shape[0] < self.config.window_size:
            # Pad with zeros if needed
            padding = np.zeros((self.config.window_size - combined_features.shape[0], combined_features.shape[1]))
            combined_features = np.vstack([padding, combined_features])
        
        return combined_features.astype(np.float32)
    
    def _extract_features(self, df: pd.DataFrame, symbol: str) -> np.ndarray:
        """
        Extract features from dataframe for observation.
        
        Args:
            df: DataFrame containing historical data
            symbol: Trading symbol
            
        Returns:
            NumPy array of features
        """
        features_list = []
        
        # OHLCV features
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                features_list.append(df[col].values)
            else:
                features_list.append(np.zeros(len(df)))
        
        # Spread feature
        if 'spread' in df.columns:
            features_list.append(df['spread'].values)
        else:
            features_list.append(np.ones(len(df)) * 0.0001)  # Default spread
        
        # Technical indicators
        if self.config.include_technical_indicators:
            indicator_columns = [col for col in df.columns if col.startswith(('rsi_', 'macd_', 'ma_', 'bb_'))]
            for col in indicator_columns[:10]:  # Limit to 10 indicators
                features_list.append(df[col].values)
            
            # Pad if fewer than 10 indicators
            pad_count = 10 - len(indicator_columns)
            for _ in range(pad_count):
                features_list.append(np.zeros(len(df)))
        
        # Order book features
        if self.config.include_order_book_features:
            # In a real implementation, these would come from the broker simulator
            # For now, we'll just use placeholders
            for _ in range(10):
                features_list.append(np.zeros(len(df)))
        
        # Market regime features
        if self.config.include_market_regime_features:
            # In a real implementation, these would be derived from market analysis
            # For now, we'll just use placeholders
            for _ in range(5):
                features_list.append(np.zeros(len(df)))
        
        # Time features
        if self.config.include_time_features:
            # Hour of day (0-23) normalized to [0, 1]
            hour_of_day = np.array([ts.hour / 23.0 for ts in df.index])
            features_list.append(hour_of_day)
            
            # Day of week (0-6) normalized to [0, 1]
            day_of_week = np.array([ts.weekday() / 6.0 for ts in df.index])
            features_list.append(day_of_week)
            
            # Placeholder for 3 more time features
            for _ in range(3):
                features_list.append(np.zeros(len(df)))
        
        # Position features
        positions = self.broker_simulator.get_positions()
        position_size = 0.0
        position_pnl = 0.0
        entry_price = 0.0
        
        if symbol in positions:
            position = positions[symbol]
            position_size = position["quantity"] / (self.config.initial_balance * self.config.leverage / 100.0)  # Normalize
            position_pnl = position["unrealized_pnl"] / self.config.initial_balance  # Normalize
            entry_price = position.get("avg_price", 0.0)
            if entry_price > 0:
                current_price = df['close'].iloc[-1] if len(df) > 0 else 0
                entry_price = (entry_price - current_price) / current_price if current_price > 0 else 0  # Normalize
        
        features_list.append(np.ones(len(df)) * position_size)
        features_list.append(np.ones(len(df)) * position_pnl)
        features_list.append(np.ones(len(df)) * entry_price)
        
        # Convert to numpy array and transpose
        features = np.array(features_list).T
        
        return features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to [-1, 1] range.
        
        Args:
            features: NumPy array of features
            
        Returns:
            Normalized features
        """
        # Skip if already normalized or empty
        if features.size == 0:
            return features
            
        # Simple min-max normalization to [-1, 1] with clipping
        # Calculate min and max for each feature
        min_vals = np.nanmin(features, axis=0)
        max_vals = np.nanmax(features, axis=0)
        
        # Replace NaN with 0
        min_vals = np.nan_to_num(min_vals)
        max_vals = np.nan_to_num(max_vals)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        # Normalize
        normalized = 2 * (features - min_vals) / range_vals - 1
        
        # Clip to ensure [-1, 1] range
        normalized = np.clip(normalized, -1, 1)
        
        return normalized
    
    def _calculate_reward(self) -> float:
        """
        Calculate the reward based on the configured reward function.
        
        Returns:
            Reward value
        """
        if len(self.account_history) < 2:
            return 0.0
            
        # Get current and previous account state
        current_state = self.account_history[-1]
        previous_state = self.account_history[-2]
        
        if self.config.reward_function == RewardType.PNL_CHANGE:
            # Simple P&L change
            return (current_state["equity"] - previous_state["equity"]) / self.config.initial_balance
            
        elif self.config.reward_function == RewardType.RISK_ADJUSTED:
            # Risk-adjusted return (Sharpe-like)
            equity_change = current_state["equity"] - previous_state["equity"]
            
            # Calculate volatility using recent equity changes (use at least 10 steps)
            history_window = min(10, len(self.account_history))
            if history_window >= 3:  # Need at least 3 points to calculate meaningful std
                equity_changes = [
                    self.account_history[i]["equity"] - self.account_history[i-1]["equity"]
                    for i in range(-history_window, 0)
                ]
                volatility = np.std(equity_changes) + 1e-6  # Avoid division by zero
                return equity_change / volatility
            else:
                return equity_change / self.config.initial_balance
                
        elif self.config.reward_function == RewardType.DRAWDOWN_PENALIZED:
            # P&L with drawdown penalty
            equity_change = current_state["equity"] - previous_state["equity"]
            
            # Calculate drawdown
            max_equity = max(state["equity"] for state in self.account_history)
            drawdown = (max_equity - current_state["equity"]) / max_equity
            
            # Penalize drawdown
            drawdown_penalty = drawdown * 2.0
            
            return (equity_change / self.config.initial_balance) - drawdown_penalty
            
        elif self.config.reward_function == RewardType.DIRECTIONAL_CORRECTNESS:
            # Reward for correct direction prediction
            reward = 0.0
            
            # Check each open position
            positions = self.broker_simulator.get_positions()
            
            for symbol, position in positions.items():
                if symbol not in self.historical_data:
                    continue
                    
                # Get current price data
                df = self.historical_data[symbol]
                if df.empty:
                    continue
                
                try:
                    idx = df.index.get_indexer([self.current_time], method='pad')[0]
                except:
                    continue
                    
                if idx < 1 or idx >= len(df):
                    continue
                
                # Get price change
                price_change = df['close'].iloc[idx] - df['close'].iloc[idx-1]
                
                # Position direction matches price movement?
                if (position["quantity"] > 0 and price_change > 0) or (position["quantity"] < 0 and price_change < 0):
                    reward += 0.1  # Small positive reward for correct direction
                else:
                    reward -= 0.1  # Small negative reward for wrong direction
            
            return reward
            
        elif self.config.reward_function == RewardType.CUSTOM and self.custom_reward_function:
            # Use custom reward function
            return self.custom_reward_function(self.account_history, self.trades, self.broker_simulator)
            
        # Default reward
        return (current_state["equity"] - previous_state["equity"]) / self.config.initial_balance
    
    def _get_step_info(self) -> Dict[str, Any]:
        """
        Get additional info for the current step.
        
        Returns:
            Dictionary of info
        """
        if not self.account_history:
            return {}
            
        current_state = self.account_history[-1]
        
        # Calculate performance metrics
        initial_equity = self.config.initial_balance
        current_equity = current_state["equity"]
        
        # Calculate returns
        returns_pct = (current_equity / initial_equity - 1) * 100
        
        # Calculate drawdown
        max_equity = max(state["equity"] for state in self.account_history)
        drawdown_pct = (max_equity - current_equity) / max_equity * 100
        
        # Get open positions
        positions = self.broker_simulator.get_positions()
        
        return {
            "step": self.current_step,
            "timestamp": self.current_time.isoformat(),
            "equity": current_equity,
            "returns_pct": returns_pct,
            "drawdown_pct": drawdown_pct,
            "margin_used": current_state["margin_used"],
            "open_positions": len(positions),
            "position_details": [
                {
                    "symbol": symbol,
                    "quantity": pos["quantity"],
                    "unrealized_pnl": pos["unrealized_pnl"]
                }
                for symbol, pos in positions.items()
            ]
        }
    
    def render(self, mode='human'):
        """Render the environment."""
        if not self.account_history:
            return
            
        current_state = self.account_history[-1]
        positions = self.broker_simulator.get_positions()
        
        print(f"\n=== Step {self.current_step} | Time: {self.current_time} ===")
        print(f"Equity: ${current_state['equity']:.2f}")
        print(f"Balance: ${current_state['balance']:.2f}")
        print(f"Unrealized P&L: ${current_state['unrealized_pnl']:.2f}")
        
        if positions:
            print("\nOpen Positions:")
            for symbol, pos in positions.items():
                print(f"  {symbol}: {pos['quantity']:.4f} lots | P&L: ${pos['unrealized_pnl']:.2f}")
        else:
            print("\nNo open positions")
    
    def close(self):
        """Close the environment and clean up resources."""
        # Nothing to clean up for now
        pass
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a performance summary for the completed episode.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.account_history:
            return {
                "returns_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "num_trades": 0,
                "win_rate": 0.0,
                "avg_profit_per_trade": 0.0,
                "avg_loss_per_trade": 0.0,
                "profit_factor": 0.0
            }
            
        # Calculate equity curve
        equity_curve = np.array([state["equity"] for state in self.account_history])
        
        # Calculate returns
        initial_equity = self.config.initial_balance
        final_equity = equity_curve[-1]
        returns_pct = (final_equity / initial_equity - 1) * 100
        
        # Calculate drawdown
        peaks = np.maximum.accumulate(equity_curve)
        drawdowns = (peaks - equity_curve) / peaks * 100
        max_drawdown = np.max(drawdowns)
        
        # Calculate Sharpe ratio (if enough data)
        daily_returns = []
        if len(equity_curve) > 1:
            daily_returns = np.diff(equity_curve) / equity_curve[:-1]
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns) + 1e-6  # Avoid division by zero
            sharpe = avg_return / std_return * np.sqrt(252)  # Annualized
        else:
            sharpe = 0.0
            
        # Analyze trades
        num_trades = len(self.trades)
        profitable_trades = 0
        unprofitable_trades = 0
        total_profit = 0.0
        total_loss = 0.0
        
        # This is simplified - in a real implementation we would track actual P&L per trade
        for i, trade in enumerate(self.trades):
            if i == len(self.trades) - 1:
                continue  # Skip the last trade if it's not closed
                
            if trade["side"] == "close":
                continue  # Skip close orders
                
            next_trade = None
            for j in range(i+1, len(self.trades)):
                if self.trades[j]["symbol"] == trade["symbol"] and self.trades[j]["side"] == "close":
                    next_trade = self.trades[j]
                    break
                    
            if next_trade:
                price_diff = next_trade["price"] - trade["price"]
                if trade["side"] == "sell":
                    price_diff = -price_diff
                    
                pnl = price_diff * trade["quantity"]
                
                if pnl > 0:
                    profitable_trades += 1
                    total_profit += pnl
                else:
                    unprofitable_trades += 1
                    total_loss += abs(pnl)
        
        # Calculate trade metrics
        win_rate = profitable_trades / num_trades if num_trades > 0 else 0.0
        avg_profit = total_profit / profitable_trades if profitable_trades > 0 else 0.0
        avg_loss = total_loss / unprofitable_trades if unprofitable_trades > 0 else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
        
        return {
            "returns_pct": returns_pct,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "avg_profit_per_trade": avg_profit,
            "avg_loss_per_trade": avg_loss,
            "profit_factor": profit_factor
        }


class EnvironmentGenerator:
    """
    Factory class for generating RL environments with different configurations.
    """
    
    @staticmethod
    def create_environment(
        market_simulator: AdvancedMarketRegimeSimulator,
        broker_simulator: ForexBrokerSimulator,
        config: EnvConfiguration = None,
        custom_reward_fn: Callable = None,
        random_seed: int = None
    ) -> ForexTradingEnvironment:
        """
        Create a forex trading environment.
        
        Args:
            market_simulator: Advanced market regime simulator
            broker_simulator: Forex broker simulator
            config: Optional environment configuration
            custom_reward_fn: Optional custom reward function
            random_seed: Optional random seed
            
        Returns:
            Configured ForexTradingEnvironment
        """
        if config is None:
            config = EnvConfiguration(symbols=["EUR/USD"])
            
        return ForexTradingEnvironment(
            market_simulator=market_simulator,
            broker_simulator=broker_simulator,
            config=config,
            custom_reward_function=custom_reward_fn,
            random_seed=random_seed
        )
    
    @staticmethod
    def create_multiasset_environment(
        symbols: List[str],
        action_type: str = "discrete",
        reward_type: RewardType = RewardType.RISK_ADJUSTED,
        window_size: int = 20,
        include_indicators: bool = True
    ) -> ForexTradingEnvironment:
        """
        Create a multi-asset trading environment.
        
        Args:
            symbols: List of trading symbols
            action_type: Type of action space ('discrete' or 'continuous')
            reward_type: Type of reward function
            window_size: Size of observation window
            include_indicators: Whether to include technical indicators
            
        Returns:
            Configured multi-asset ForexTradingEnvironment
        """
        # Create simulators
        broker_sim = ForexBrokerSimulator(balance=10000.0, leverage=100.0)
        market_sim = AdvancedMarketRegimeSimulator(broker_simulator=broker_sim)
        
        # Create configuration
        config = EnvConfiguration(
            symbols=symbols,
            action_type=action_type,
            reward_function=reward_type,
            window_size=window_size,
            include_technical_indicators=include_indicators
        )
        
        return EnvironmentGenerator.create_environment(
            market_simulator=market_sim,
            broker_simulator=broker_sim,
            config=config
        )
    
    @staticmethod
    def load_environment_config(config_path: str) -> ForexTradingEnvironment:
        """
        Load environment configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configured ForexTradingEnvironment
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            
        # Create simulators
        broker_sim = ForexBrokerSimulator(
            balance=config_data.get("initial_balance", 10000.0),
            leverage=config_data.get("leverage", 100.0)
        )
        
        market_sim = AdvancedMarketRegimeSimulator(broker_simulator=broker_sim)
        
        # Create configuration
        symbols = config_data.get("symbols", ["EUR/USD"])
        reward_type = RewardType(config_data.get("reward_type", "risk_adjusted"))
        
        config = EnvConfiguration(
            symbols=symbols,
            timeframe=config_data.get("timeframe", "1h"),
            initial_balance=config_data.get("initial_balance", 10000.0),
            leverage=config_data.get("leverage", 100.0),
            action_type=config_data.get("action_type", "discrete"),
            reward_function=reward_type,
            max_episode_steps=config_data.get("max_episode_steps", 1000),
            window_size=config_data.get("window_size", 20),
            include_technical_indicators=config_data.get("include_indicators", True)
        )
        
        return EnvironmentGenerator.create_environment(
            market_simulator=market_sim,
            broker_simulator=broker_sim,
            config=config
        )
