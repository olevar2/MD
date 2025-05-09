"""
State Representation Components for Reinforcement Learning Environments

This package contains modular state representation components for RL environments:
- Observation space definitions
- Feature extractors for different data types
- State representation utilities
"""

from .observation_space import ObservationSpaceBuilder
from .feature_extractors import FeatureExtractor
from .state_representation import StateRepresentation

__all__ = [
    "ObservationSpaceBuilder",
    "FeatureExtractor",
    "StateRepresentation",
]