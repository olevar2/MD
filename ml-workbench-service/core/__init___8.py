"""
Reinforcement Learning Environment Package

This package contains modular components for reinforcement learning environments:
- Base environment definitions
- Reward calculation components
- State representation components

The modular design allows for easy customization and extension of RL environments.
"""

from .base_environment import BaseRLEnvironment
from .forex_environment import ForexTradingEnvironment

__all__ = [
    "BaseRLEnvironment",
    "ForexTradingEnvironment",
]