"""
RLAgent Implementation

This module implements a Reinforcement Learning agent for forex trading that learns optimal
trade execution and dynamic risk management strategies through interaction with the market.
"""

import os
import json
import logging
import random
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class ForexTradingEnvironment:
    """
    A Gym-compatible environment for forex trading.
    
    This environment simulates a forex trading market where the agent can
    take actions (buy, sell, hold) and receive rewards based on profit & loss,
    transaction costs, and risk metrics.
    
    Attributes:
        data: Market data used for the simulation
        state_size: Dimension of the state space
        action_size: Dimension of the action space
        current_step: Current position in the data
        max_steps: Maximum number of steps in an episode
        transaction_cost: Cost of executing a trade
        risk_penalty: Scaling factor for risk penalties
        initial_balance: Starting account balance
    """
    
    def __init__(
        self, 
        data: pd.DataFrame,
        state_size: int,
        action_size: int = 3,  # Buy, Sell, Hold
        transaction_cost: float = 0.0001,
        risk_penalty: float = 0.1,
        initial_balance: float = 10000.0,
        max_steps: Optional[int] = None,
        random_start: bool = True
    ):
        """
        Initialize the trading environment.
        
        Args:
            data: Dataframe containing market data (OHLCV, indicators, etc.)
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            transaction_cost: Cost per transaction as a fraction of trade value
            risk_penalty: Coefficient for risk penalty in the reward function
            initial_balance: Starting account balance
            max_steps: Maximum number of steps in an episode (defaults to data length)
            random_start: Whether to start episodes at random positions
        """
        self.data = data
        self.state_size = state_size
        self.action_size = action_size
        self.transaction_cost = transaction_cost
        self.risk_penalty = risk_penalty
        self.initial_balance = initial_balance
        self.max_steps = max_steps if max_steps is not None else len(data) - 1
        self.random_start = random_start
        
        # Current state variables
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.entry_price = 0.0
        self.total_reward = 0.0
        self.done = False
        
        # Track metrics for evaluation
        self.trades = []
        self.returns = []
        self.positions = []
        
    def reset(self) -> np.ndarray:
        """
        Reset the environment to start a new episode.
        
        Returns:
            Initial state observation
        """
        # Reset position in the data
        if self.random_start:
            self.current_step = random.randint(0, len(self.data) - self.max_steps - 1)
        else:
            self.current_step = 0
            
        # Reset trading variables
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.total_reward = 0.0
        self.done = False
        
        # Reset tracking variables
        self.trades = []
        self.returns = []
        self.positions = []
        
        # Return the initial state
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0: Buy, 1: Sell, 2: Hold)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Ensure action is valid
        assert 0 <= action < self.action_size
        
        # Get current price data
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        
        # Process the action
        reward = 0.0
        info = {}
        
        # Calculate price change
        price_change = (next_price - current_price) / current_price
        
        # Process based on action and current position
        if action == 0:  # Buy
            if self.position <= 0:  # If not already long
                # Close any existing short position
                if self.position == -1:
                    trade_return = self.entry_price / current_price - 1
                    self.returns.append(trade_return)
                    trade_reward = self.balance * trade_return
                    reward += trade_reward
                    
                    # Record the trade
                    self.trades.append({
                        'entry_time': self.data.index[self.current_step - len(self.positions)],
                        'exit_time': self.data.index[self.current_step],
                        'entry_price': self.entry_price,
                        'exit_price': current_price,
                        'position': -1,
                        'return': trade_return,
                        'reward': trade_reward
                    })
                
                # Enter a new long position
                self.position = 1
                self.entry_price = current_price
                # Apply transaction cost
                reward -= self.balance * self.transaction_cost
                
        elif action == 1:  # Sell
            if self.position >= 0:  # If not already short
                # Close any existing long position
                if self.position == 1:
                    trade_return = current_price / self.entry_price - 1
                    self.returns.append(trade_return)
                    trade_reward = self.balance * trade_return
                    reward += trade_reward
                    
                    # Record the trade
                    self.trades.append({
                        'entry_time': self.data.index[self.current_step - len(self.positions)],
                        'exit_time': self.data.index[self.current_step],
                        'entry_price': self.entry_price,
                        'exit_price': current_price,
                        'position': 1,
                        'return': trade_return,
                        'reward': trade_reward
                    })
                
                # Enter a new short position
                self.position = -1
                self.entry_price = current_price
                # Apply transaction cost
                reward -= self.balance * self.transaction_cost
        
        # Otherwise action is Hold (2), do nothing
        
        # Update balance based on current position and market movement
        if self.position == 1:  # Long position
            unrealized_return = price_change
            self.balance *= (1 + unrealized_return)
        elif self.position == -1:  # Short position
            unrealized_return = -price_change
            self.balance *= (1 + unrealized_return)
        
        # Apply risk penalty based on volatility
        volatility = self.data.iloc[self.current_step].get('volatility', 
                                                           abs(self.data.iloc[self.current_step]['high'] - 
                                                               self.data.iloc[self.current_step]['low']) / 
                                                           self.data.iloc[self.current_step]['close'])
        risk_penalty = volatility * abs(self.position) * self.risk_penalty
        reward -= risk_penalty
        
        # Record the current position
        self.positions.append(self.position)
        
        # Update current step and check if episode is done
        self.current_step += 1
        if self.current_step >= len(self.data) - 1 or self.current_step >= self.max_steps:
            self.done = True
            
            # Close any open position at the end
            if self.position != 0:
                final_price = self.data.iloc[self.current_step]['close']
                if self.position == 1:
                    trade_return = final_price / self.entry_price - 1
                else:  # position == -1
                    trade_return = self.entry_price / final_price - 1
                    
                self.returns.append(trade_return)
                trade_reward = self.balance * trade_return
                reward += trade_reward
                
                # Record the final trade
                self.trades.append({
                    'entry_time': self.data.index[self.current_step - len(self.positions)],
                    'exit_time': self.data.index[self.current_step],
                    'entry_price': self.entry_price,
                    'exit_price': final_price,
                    'position': self.position,
                    'return': trade_return,
                    'reward': trade_reward
                })
        
        # Update total reward
        self.total_reward += reward
        
        # Prepare info dictionary
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_reward': self.total_reward,
            'num_trades': len(self.trades)
        }
        
        # Get next observation
        next_state = self._get_observation()
        
        return next_state, reward, self.done, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current state observation.
        
        Returns:
            Array representing the current state
        """
        # Extract relevant features from the current data point
        market_features = self.data.iloc[self.current_step].values
        
        # Add position state to the observation
        position_onehot = np.zeros(3)  # -1, 0, 1 -> one-hot encoding
        position_onehot[self.position + 1] = 1
        
        # Combine market features and position
        observation = np.concatenate([market_features, position_onehot])
        
        return observation
    
    def render(self, mode: str = 'human') -> None:
        """
        Render the environment state.
        
        Args:
            mode: Rendering mode ('human' for console output)
        """
        if mode == 'human':
            current_date = self.data.index[self.current_step]
            current_price = self.data.iloc[self.current_step]['close']
            position_map = {-1: 'SHORT', 0: 'NEUTRAL', 1: 'LONG'}
            
            print(f"Date: {current_date}")
            print(f"Price: {current_price:.5f}")
            print(f"Position: {position_map[self.position]}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Total Reward: {self.total_reward:.2f}")
            print(f"Number of Trades: {len(self.trades)}")
            print("-" * 50)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate trading statistics for the episode.
        
        Returns:
            Dictionary of trading statistics
        """
        if not self.trades:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'avg_trade_return': 0.0,
                'max_drawdown': 0.0,
                'num_trades': 0
            }
        
        # Calculate returns
        total_return = (self.balance / self.initial_balance) - 1
        returns = np.array(self.returns)
        
        # Calculate Sharpe ratio (annualized)
        sharpe_ratio = 0.0
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
            
        # Calculate win rate
        win_rate = np.sum(returns > 0) / len(returns)
        
        # Calculate average trade return
        avg_trade_return = np.mean(returns)
        
        # Calculate maximum drawdown
        peak = self.initial_balance
        max_drawdown = 0
        balance_history = [self.initial_balance]
        for r in returns:
            balance = balance_history[-1] * (1 + r)
            balance_history.append(balance)
            peak = max(peak, balance)
            drawdown = (peak - balance) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades),
            'trades': self.trades
        }


class ReplayBuffer:
    """
    Experience replay buffer for RL agent training.
    
    Stores and samples transitions (state, action, reward, next_state, done) for
    training the agent with experience replay.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), 
                np.array(rewards, dtype=np.float32), 
                np.array(next_states), np.array(dones, dtype=np.uint8))
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)


class RLAgent:
    """
    Reinforcement Learning agent for forex trading.
    
    Implements a Deep Q-Network (DQN) or Policy Gradient approach to learn
    optimal trading strategies through market interaction.
    
    Attributes:
        state_size: Dimension of the state space
        action_size: Dimension of the action space
        memory: Experience replay buffer
        model: Policy or Q-network model
        target_model: Target network for stable Q-learning
        gamma: Discount factor for future rewards
        epsilon: Exploration rate
        epsilon_decay: Rate of exploration decay
        epsilon_min: Minimum exploration rate
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        memory_capacity: int = 10000,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        learning_rate: float = 0.001,
        model_type: str = "dqn"  # "dqn" or "ppo"
    ):
        """
        Initialize the RL agent.
        
        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space (Buy, Sell, Hold)
            memory_capacity: Size of the replay buffer
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Rate of exploration decay
            epsilon_min: Minimum exploration rate
            learning_rate: Learning rate for the optimizer
            model_type: Type of RL algorithm to use ("dqn" or "ppo")
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(memory_capacity)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.model_type = model_type
        
        # Build models
        self.model = self._build_model()
        
        # For DQN, create a target model for stability
        if model_type == "dqn":
            self.target_model = self._build_model()
            self.update_target_model()
            
        # For tracking
        self.train_steps = 0
        
    def _build_model(self) -> Model:
        """
        Build the neural network model.
        
        Returns:
            Compiled TensorFlow model
        """
        if self.model_type == "dqn":
            # Build a Q-network for DQN
            model = Sequential()
            model.add(Dense(64, input_dim=self.state_size, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            return model
        elif self.model_type == "ppo":
            # Build an actor-critic architecture for PPO
            # Actor network (policy)
            state_input = Input(shape=(self.state_size,))
            dense1 = Dense(64, activation='relu')(state_input)
            dense2 = Dense(64, activation='relu')(dense1)
            
            # Policy output (actor)
            action_probs = Dense(self.action_size, activation='softmax')(dense2)
            
            # Value output (critic)
            value = Dense(1, activation='linear')(dense2)
            
            # Create model
            model = Model(inputs=state_input, outputs=[action_probs, value])
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss=['categorical_crossentropy', 'mse']
            )
            return model
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
    
    def update_target_model(self) -> None:
        """Update the target model for DQN with weights from the main model."""
        if self.model_type == "dqn":
            self.target_model.set_weights(self.model.get_weights())
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose an action based on the current state.
        
        Args:
            state: Current state observation
            training: Whether the agent is training (enables exploration)
            
        Returns:
            Selected action
        """
        if training and np.random.rand() <= self.epsilon:
            # Exploration: select random action
            return random.randrange(self.action_size)
        
        if self.model_type == "dqn":
            # Exploitation: select best action based on Q-values
            act_values = self.model.predict(np.array([state]), verbose=0)
            return np.argmax(act_values[0])
        elif self.model_type == "ppo":
            # Get action probabilities from policy
            action_probs, _ = self.model.predict(np.array([state]), verbose=0)
            # Sample from the probability distribution
            return np.random.choice(self.action_size, p=action_probs[0])
    
    def remember(self, state, action, reward, next_state, done) -> None:
        """
        Store a transition in the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.memory.add(state, action, reward, next_state, done)
    
    def replay(self, batch_size: int) -> float:
        """
        Train the agent using experience replay.
        
        Args:
            batch_size: Number of samples to train on
            
        Returns:
            Training loss
        """
        if len(self.memory) < batch_size:
            return 0.0
        
        if self.model_type == "dqn":
            # Sample a batch from memory
            states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
            
            # DQN training
            targets = self.model.predict(states, verbose=0)
            next_q_values = self.target_model.predict(next_states, verbose=0)
            
            for i in range(batch_size):
                if dones[i]:
                    targets[i, actions[i]] = rewards[i]
                else:
                    targets[i, actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values[i])
            
            # Train the model
            history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
            
            # Update epsilon for exploration
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            # Update target model periodically
            self.train_steps += 1
            if self.train_steps % 100 == 0:
                self.update_target_model()
                
            return history.history['loss'][0]
        
        elif self.model_type == "ppo":
            # PPO requires more complex implementation that is beyond this simplified example
            # Would need to track advantages, implement clipping, etc.
            logger.warning("PPO training not fully implemented in this simplified version")
            return 0.0
    
    def save(self, path: str) -> None:
        """
        Save the agent's models and parameters.
        
        Args:
            path: Directory path to save the agent
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save the model
        self.model.save(os.path.join(path, "model.h5"))
        
        # For DQN, also save target model
        if self.model_type == "dqn":
            self.target_model.save(os.path.join(path, "target_model.h5"))
        
        # Save parameters
        params = {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "learning_rate": self.learning_rate,
            "model_type": self.model_type,
            "train_steps": self.train_steps,
            "date_saved": datetime.now().isoformat()
        }
        
        with open(os.path.join(path, "params.json"), "w") as f:
            json.dump(params, f)
            
        logger.info(f"Saved RLAgent to {path}")
    
    @classmethod
    def load(cls, path: str) -> "RLAgent":
        """
        Load a saved agent from disk.
        
        Args:
            path: Path to the saved agent directory
            
        Returns:
            Loaded RLAgent instance
        """
        # Load parameters
        with open(os.path.join(path, "params.json"), "r") as f:
            params = json.load(f)
            
        # Create agent instance
        agent = cls(
            state_size=params["state_size"],
            action_size=params["action_size"],
            gamma=params["gamma"],
            epsilon=params["epsilon"],
            epsilon_decay=params["epsilon_decay"],
            epsilon_min=params["epsilon_min"],
            learning_rate=params["learning_rate"],
            model_type=params["model_type"]
        )
        
        # Load models
        agent.model = tf.keras.models.load_model(os.path.join(path, "model.h5"))
        
        if params["model_type"] == "dqn":
            agent.target_model = tf.keras.models.load_model(os.path.join(path, "target_model.h5"))
            
        agent.train_steps = params["train_steps"]
        
        logger.info(f"Loaded RLAgent from {path}")
        return agent
    
    def train(
        self, 
        env: ForexTradingEnvironment, 
        episodes: int = 100, 
        batch_size: int = 32,
        max_steps_per_episode: Optional[int] = None,
        render_every: int = 10,
        save_path: Optional[str] = None,
        callback = None
    ) -> Dict[str, List[float]]:
        """
        Train the agent on the given environment.
        
        Args:
            env: Trading environment
            episodes: Number of episodes to train
            batch_size: Batch size for experience replay
            max_steps_per_episode: Maximum steps per episode (None for env default)
            render_every: How often to render the environment (0 to disable)
            save_path: Where to save the model (None to disable saving)
            callback: Optional callback function called after each episode
            
        Returns:
            Dictionary with training metrics
        """
        scores = []
        avg_scores = []
        balances = []
        sharpe_ratios = []
        win_rates = []
        
        for episode in range(episodes):
            # Reset environment
            state = env.reset()
            score = 0
            done = False
            step = 0
            
            while not done:
                # Select and perform an action
                action = self.act(state)
                next_state, reward, done, info = env.step(action)
                
                # Store transition in memory
                self.remember(state, action, reward, next_state, done)
                
                # Move to the next state
                state = next_state
                score += reward
                step += 1
                
                # Train on a batch of transitions
                if len(self.memory) >= batch_size:
                    self.replay(batch_size)
                
                # Check if episode should end early
                if max_steps_per_episode and step >= max_steps_per_episode:
                    break
            
            # Get episode statistics
            stats = env.get_statistics()
            
            # Save metrics
            scores.append(score)
            balances.append(info['balance'])
            sharpe_ratios.append(stats['sharpe_ratio'])
            win_rates.append(stats['win_rate'])
            
            # Calculate rolling average score
            avg_score = np.mean(scores[-100:])  # Average over last 100 episodes
            avg_scores.append(avg_score)
            
            # Log progress
            logger.info(f"Episode: {episode+1}/{episodes} | "
                       f"Score: {score:.2f} | "
                       f"Avg Score: {avg_score:.2f} | "
                       f"Epsilon: {self.epsilon:.4f} | "
                       f"Balance: ${info['balance']:.2f} | "
                       f"Trades: {stats['num_trades']} | "
                       f"Win Rate: {stats['win_rate']:.2f} | "
                       f"Sharpe: {stats['sharpe_ratio']:.2f}")
            
            # Render environment if needed
            if render_every > 0 and episode % render_every == 0:
                env.render()
                
            # Save model periodically
            if save_path and (episode + 1) % 10 == 0:
                episode_save_path = os.path.join(save_path, f"episode_{episode+1}")
                self.save(episode_save_path)
                
            # Call callback if provided
            if callback:
                callback(episode, stats, self)
        
        # Save final model
        if save_path:
            self.save(os.path.join(save_path, "final"))
            
        return {
            'scores': scores,
            'avg_scores': avg_scores,
            'balances': balances,
            'sharpe_ratios': sharpe_ratios,
            'win_rates': win_rates
        }
