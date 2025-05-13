"""
Reinforcement Learning Interfaces Module

This module provides interfaces for reinforcement learning components used across services,
helping to break circular dependencies between services.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime


class RLModelType(str, Enum):
    """Types of reinforcement learning models."""
    PPO = "ppo"
    A2C = "a2c"
    DQN = "dqn"
    SAC = "sac"
    TD3 = "td3"
    CUSTOM = "custom"


class RLConfidenceLevel(str, Enum):
    """Confidence levels for RL model predictions."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class IRLModel(ABC):
    """Interface for reinforcement learning models."""
    
    @abstractmethod
    def predict(self, state: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        Generate a prediction from the model.
        
        Args:
            state: The current state representation
            
        Returns:
            Tuple of (action, metadata)
        """
        pass
    
    @abstractmethod
    def get_confidence(self, state: Dict[str, Any], action: Any) -> float:
        """
        Get the confidence level for a prediction.
        
        Args:
            state: The current state representation
            action: The predicted action
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    def update(self, state: Dict[str, Any], action: Any, reward: float, next_state: Dict[str, Any], done: bool) -> None:
        """
        Update the model with a new experience.
        
        Args:
            state: The starting state
            action: The action taken
            reward: The reward received
            next_state: The resulting state
            done: Whether the episode is done
        """
        pass


class IRLOptimizer(ABC):
    """Interface for RL-based parameter optimizers."""
    
    @abstractmethod
    async def optimize_parameters(
        self,
        parameter_type: str,
        current_values: Dict[str, Any],
        context: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize parameters using RL insights.
        
        Args:
            parameter_type: Type of parameters to optimize
            current_values: Current parameter values
            context: Contextual information for optimization
            constraints: Optional constraints on parameter values
            
        Returns:
            Optimized parameter values
        """
        pass
    
    @abstractmethod
    def get_optimization_confidence(self) -> float:
        """
        Get the confidence level of the last optimization.
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    async def update_models(self) -> bool:
        """
        Update the underlying RL models.
        
        Returns:
            Success status of the update
        """
        pass


class IRLEnvironment(ABC):
    """Interface for reinforcement learning environments."""
    
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to an initial state.
        
        Returns:
            Initial observation
        """
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        pass
    
    @abstractmethod
    def render(self, mode: str = 'human') -> Optional[Any]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            Rendering result, if any
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the environment and release resources."""
        pass


class IRiskParameterOptimizer(ABC):
    """Interface for risk parameter optimizers."""
    
    @abstractmethod
    async def optimize_risk_parameters(
        self,
        symbol: str,
        current_market_data: Any,
        market_regime: Any,
        current_portfolio_state: Dict[str, Any],
        prediction_confidence: Optional[float] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Generate optimized risk parameters using RL insights.
        
        Args:
            symbol: Trading symbol
            current_market_data: Recent market data for context
            market_regime: Detected market regime
            current_portfolio_state: Current positions and exposure
            prediction_confidence: Optional confidence score from prediction model
            
        Returns:
            Tuple of (optimized risk parameters, metadata about adjustments)
        """
        pass
    
    @abstractmethod
    async def update_rl_models(self) -> bool:
        """
        Update the RL models with the latest training data.
        
        Returns:
            Success status of the update
        """
        pass
    
    @abstractmethod
    def get_adjustment_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of parameter adjustments.
        
        Returns:
            List of adjustment records
        """
        pass
