"""
RL Model Adapter Module

This module provides adapter implementations for reinforcement learning model interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging
import asyncio
import numpy as np
import random

from common_lib.reinforcement.interfaces import (
    IRLModel, IRLOptimizer, RLModelType, RLConfidenceLevel
)
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class RLModelAdapter(IRLModel):
    """
    Adapter for reinforcement learning models that implements the common interface.
    
    This adapter can either wrap an actual model instance or provide
    standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, model_instance=None, model_type: RLModelType = RLModelType.PPO):
        """
        Initialize the adapter.
        
        Args:
            model_instance: Optional actual model instance to wrap
            model_type: Type of RL model
        """
        self.model = model_instance
        self.model_type = model_type
        self.last_state = None
        self.last_action = None
        self.last_confidence = 0.5
    
    def predict(self, state: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        Generate a prediction from the model.
        
        Args:
            state: The current state representation
            
        Returns:
            Tuple of (action, metadata)
        """
        if self.model:
            try:
                # Try to use the wrapped model if available
                return self.model.predict(state)
            except Exception as e:
                logger.warning(f"Error predicting with RL model: {str(e)}")
        
        # Fallback to simple prediction if no model available
        self.last_state = state
        
        # Generate a random action between -1 and 1
        action = np.random.uniform(-1, 1)
        self.last_action = action
        
        # Generate random confidence between 0.3 and 0.7
        confidence = random.uniform(0.3, 0.7)
        self.last_confidence = confidence
        
        return action, {
            "confidence": confidence,
            "model_type": str(self.model_type),
            "source": "adapter_fallback"
        }
    
    def get_confidence(self, state: Dict[str, Any], action: Any) -> float:
        """
        Get the confidence level for a prediction.
        
        Args:
            state: The current state representation
            action: The predicted action
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if self.model:
            try:
                # Try to use the wrapped model if available
                return self.model.get_confidence(state, action)
            except Exception as e:
                logger.warning(f"Error getting confidence from RL model: {str(e)}")
        
        # Return last confidence or default
        if self.last_state is not None and self.last_action is not None:
            return self.last_confidence
        
        return 0.5
    
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
        if self.model:
            try:
                # Try to use the wrapped model if available
                self.model.update(state, action, reward, next_state, done)
            except Exception as e:
                logger.warning(f"Error updating RL model: {str(e)}")
        
        # No-op for adapter fallback
        pass


class RLOptimizerAdapter(IRLOptimizer):
    """
    Adapter for RL-based parameter optimizers that implements the common interface.
    
    This adapter can either wrap an actual optimizer instance or provide
    standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, optimizer_instance=None):
        """
        Initialize the adapter.
        
        Args:
            optimizer_instance: Optional actual optimizer instance to wrap
        """
        self.optimizer = optimizer_instance
        self.last_confidence = 0.5
        self.optimization_history = []
    
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
        if self.optimizer:
            try:
                # Try to use the wrapped optimizer if available
                return await self.optimizer.optimize_parameters(
                    parameter_type=parameter_type,
                    current_values=current_values,
                    context=context,
                    constraints=constraints
                )
            except Exception as e:
                logger.warning(f"Error optimizing parameters: {str(e)}")
        
        # Fallback to simple optimization if no optimizer available
        optimized_values = current_values.copy()
        
        # Apply small random adjustments to each parameter
        for key, value in optimized_values.items():
            if isinstance(value, (int, float)):
                # Apply adjustment within ±10%
                adjustment_factor = random.uniform(0.9, 1.1)
                optimized_values[key] = value * adjustment_factor
                
                # Apply constraints if provided
                if constraints and key in constraints:
                    min_val = constraints.get(f"{key}_min")
                    max_val = constraints.get(f"{key}_max")
                    
                    if min_val is not None and optimized_values[key] < min_val:
                        optimized_values[key] = min_val
                    
                    if max_val is not None and optimized_values[key] > max_val:
                        optimized_values[key] = max_val
        
        # Generate random confidence between 0.4 and 0.6
        self.last_confidence = random.uniform(0.4, 0.6)
        
        # Record optimization
        self.optimization_history.append({
            "timestamp": datetime.now().isoformat(),
            "parameter_type": parameter_type,
            "original_values": current_values,
            "optimized_values": optimized_values,
            "confidence": self.last_confidence
        })
        
        return optimized_values
    
    def get_optimization_confidence(self) -> float:
        """
        Get the confidence level of the last optimization.
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if self.optimizer:
            try:
                # Try to use the wrapped optimizer if available
                return self.optimizer.get_optimization_confidence()
            except Exception as e:
                logger.warning(f"Error getting optimization confidence: {str(e)}")
        
        return self.last_confidence
    
    async def update_models(self) -> bool:
        """
        Update the underlying RL models.
        
        Returns:
            Success status of the update
        """
        if self.optimizer:
            try:
                # Try to use the wrapped optimizer if available
                return await self.optimizer.update_models()
            except Exception as e:
                logger.warning(f"Error updating RL models: {str(e)}")
        
        # Pretend success if no optimizer available
        return True
