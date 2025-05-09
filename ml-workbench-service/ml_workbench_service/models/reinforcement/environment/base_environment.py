"""
Base Reinforcement Learning Environment

This module provides the base class for all reinforcement learning environments
in the forex trading platform.
"""

import gym
from gym import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import logging

from core_foundations.utils.logger import get_logger
from common_lib.reinforcement.interfaces import IRLEnvironment

logger = get_logger(__name__)


class BaseRLEnvironment(gym.Env, ABC):
    """
    Abstract base class for all reinforcement learning environments.
    
    This class defines the common interface and functionality for all RL environments
    in the forex trading platform, following the OpenAI Gym interface.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 episode_timesteps: int = 1000,
                 random_episode_start: bool = True,
                 observation_normalization: bool = True):
        """
        Initialize the base RL environment.
        
        Args:
            episode_timesteps: Maximum timesteps per episode
            random_episode_start: Whether to start episodes at random positions
            observation_normalization: Whether to normalize observations
        """
        super(BaseRLEnvironment, self).__init__()
        
        self.max_episode_steps = episode_timesteps
        self.random_episode_start = random_episode_start
        self.observation_normalization = observation_normalization
        
        # Internal state
        self.current_step = 0
        self.current_timestamp = None
        self.state_history = []
        self.reward_history = []
        
        # These will be set by child classes
        self.observation_space = None
        self.action_space = None
    
    @abstractmethod
    def _setup_spaces(self):
        """Set up the observation and action spaces."""
        pass
    
    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Get the current observation based on the environment state."""
        pass
    
    @abstractmethod
    def _take_action(self, action: np.ndarray) -> Tuple[float, Dict]:
        """
        Execute the action in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            reward: The reward for this action
            info: Additional information
        """
        pass
    
    @abstractmethod
    def _calculate_reward(self) -> float:
        """Calculate the reward based on the current state and action."""
        pass
    
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
        self.current_step += 1
        
        # Execute the action and get reward
        reward, info = self._take_action(action)
        
        # Check if episode is done
        done = self.current_step >= self.max_episode_steps
        
        # Get the next observation
        observation = self._get_observation()
        
        # Store state for history
        self.state_history.append({
            'timestamp': self.current_timestamp,
            'observation': observation,
            'action': action,
            'reward': reward,
            'done': done,
            'info': info
        })
        
        # Store reward for history
        self.reward_history.append(reward)
        
        return observation, reward, done, info
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to start a new episode.
        
        Returns:
            observation: The initial observation
        """
        # Reset environment state
        self.current_step = 0
        self.state_history = []
        self.reward_history = []
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode != 'human':
            return
        
        print(f"Step: {self.current_step}, Time: {self.current_timestamp}")
        print("-" * 50)
    
    def close(self):
        """Clean up resources."""
        pass
    
    def seed(self, seed=None):
        """
        Set random seed.
        
        Args:
            seed: Random seed
            
        Returns:
            List containing the seed
        """
        return [seed]
    
    def get_episode_summary(self) -> Dict:
        """
        Get summary statistics for the completed episode.
        
        Returns:
            Dictionary with episode summary statistics
        """
        return {
            "total_steps": self.current_step,
            "total_reward": sum(self.reward_history),
            "mean_reward": np.mean(self.reward_history) if self.reward_history else 0.0,
            "min_reward": np.min(self.reward_history) if self.reward_history else 0.0,
            "max_reward": np.max(self.reward_history) if self.reward_history else 0.0,
        }