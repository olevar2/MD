"""
Reinforcement Learning Agent Module

This module provides implementations of reinforcement learning agents for forex trading,
optimizing execution parameters and dynamic risk control.
"""

# Import for backward compatibility
from .enhanced_rl_env_compat import EnhancedForexTradingEnv

# Import new modular components
from .environment import (
    BaseRLEnvironment,
    ForexTradingEnvironment,
)

from .environment.reward import (
    RewardComponent,
    RiskAdjustedReward,
    PnLReward,
    CustomReward,
)

from .environment.state import (
    ObservationSpaceBuilder,
    FeatureExtractor,
    StateRepresentation,
)

__all__ = [
    # Backward compatibility
    "EnhancedForexTradingEnv",
    
    # New modular components
    "BaseRLEnvironment",
    "ForexTradingEnvironment",
    "RewardComponent",
    "RiskAdjustedReward",
    "PnLReward",
    "CustomReward",
    "ObservationSpaceBuilder",
    "FeatureExtractor",
    "StateRepresentation",
]