"""
RL Environment Adapter Module

This module provides adapter implementations for reinforcement learning environment interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import random
import logging
from datetime import datetime, timedelta

from common_lib.reinforcement.interfaces import IRLEnvironment
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class RLEnvironmentAdapter(IRLEnvironment):
    """
    Adapter for reinforcement learning environments that implements the common interface.
    
    This adapter can either wrap an actual environment instance or provide
    standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, environment_instance=None):
        """
        Initialize the adapter.
        
        Args:
            environment_instance: Optional actual environment instance to wrap
        """
        self.env = environment_instance
        self.current_step = 0
        self.max_steps = 1000
        self.observation_dim = 10
        self.current_state = np.zeros(self.observation_dim)
        self.current_position = 0.0
        self.account_balance = 10000.0
        self.last_price = 1.0
        self.price_history = [1.0]
        self.reward_history = []
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to an initial state.
        
        Returns:
            Initial observation
        """
        if self.env:
            try:
                # Try to use the wrapped environment if available
                return self.env.reset()
            except Exception as e:
                logger.warning(f"Error resetting RL environment: {str(e)}")
        
        # Fallback to simple reset if no environment available
        self.current_step = 0
        self.current_position = 0.0
        self.account_balance = 10000.0
        self.last_price = 1.0
        self.price_history = [1.0]
        self.reward_history = []
        
        # Generate random initial state
        self.current_state = np.random.normal(0, 1, self.observation_dim)
        
        return {
            "observation": self.current_state,
            "account_balance": self.account_balance,
            "current_position": self.current_position,
            "current_step": self.current_step
        }
    
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.env:
            try:
                # Try to use the wrapped environment if available
                return self.env.step(action)
            except Exception as e:
                logger.warning(f"Error stepping RL environment: {str(e)}")
        
        # Fallback to simple step if no environment available
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Generate price movement
        price_change = np.random.normal(0, 0.002)
        new_price = self.last_price * (1 + price_change)
        self.price_history.append(new_price)
        
        # Calculate PnL
        position_change = float(action)
        old_position = self.current_position
        self.current_position = position_change
        
        # Calculate reward based on position and price change
        position_pnl = old_position * (new_price - self.last_price) * 10000
        self.account_balance += position_pnl
        
        # Apply trading costs
        trading_cost = abs(self.current_position - old_position) * 0.0001 * 10000
        self.account_balance -= trading_cost
        
        # Calculate reward
        reward = position_pnl - trading_cost
        self.reward_history.append(reward)
        
        # Update price
        self.last_price = new_price
        
        # Generate new state
        self.current_state = np.random.normal(0, 1, self.observation_dim)
        
        # Add price history to state
        history_len = min(5, len(self.price_history))
        recent_prices = self.price_history[-history_len:]
        for i in range(history_len):
            if i < self.observation_dim:
                self.current_state[i] = recent_prices[i] / self.price_history[0] - 1.0
        
        observation = {
            "observation": self.current_state,
            "account_balance": self.account_balance,
            "current_position": self.current_position,
            "current_step": self.current_step
        }
        
        info = {
            "price": new_price,
            "position_pnl": position_pnl,
            "trading_cost": trading_cost,
            "total_reward": sum(self.reward_history)
        }
        
        return observation, reward, done, info
    
    def render(self, mode: str = 'human') -> Optional[Any]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            Rendering result, if any
        """
        if self.env:
            try:
                # Try to use the wrapped environment if available
                return self.env.render(mode)
            except Exception as e:
                logger.warning(f"Error rendering RL environment: {str(e)}")
        
        # Simple text rendering
        if mode == 'human':
            status = (
                f"Step: {self.current_step}, "
                f"Balance: ${self.account_balance:.2f}, "
                f"Position: {self.current_position:.2f}, "
                f"Price: {self.last_price:.5f}, "
                f"Reward: {self.reward_history[-1] if self.reward_history else 0:.2f}"
            )
            print(status)
            return status
        
        return None
    
    def close(self) -> None:
        """Close the environment and release resources."""
        if self.env:
            try:
                # Try to use the wrapped environment if available
                self.env.close()
            except Exception as e:
                logger.warning(f"Error closing RL environment: {str(e)}")
        
        # No-op for adapter fallback
        pass
