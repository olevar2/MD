"""
Base Reward Components for Reinforcement Learning

This module provides the base classes for reward components in RL environments.
"""

from typing import List, Callable, Any
from dataclasses import dataclass, field


@dataclass
class RewardComponent:
    """
    A component of the reward function with weight and function.
    
    This class represents a single component of a composite reward function,
    with a weight to control its contribution to the overall reward.
    """
    name: str
    weight: float
    function: Callable
    description: str = ""
    enabled: bool = True
    history: List[float] = field(default_factory=list)
    
    def calculate(self, *args, **kwargs) -> float:
        """
        Calculate this reward component's value.
        
        Args:
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Weighted reward value
        """
        if not self.enabled:
            return 0.0
        
        value = self.function(*args, **kwargs)
        self.history.append(value)
        return value * self.weight


class CompositeReward:
    """
    A composite reward function made up of multiple components.
    
    This class manages a collection of reward components and calculates
    the total reward by summing their weighted contributions.
    """
    
    def __init__(self, components: List[RewardComponent] = None):
        """
        Initialize the composite reward.
        
        Args:
            components: List of reward components
        """
        self.components = components or []
    
    def add_component(self, component: RewardComponent):
        """
        Add a reward component.
        
        Args:
            component: Reward component to add
        """
        self.components.append(component)
    
    def remove_component(self, name: str):
        """
        Remove a reward component by name.
        
        Args:
            name: Name of the component to remove
        """
        self.components = [c for c in self.components if c.name != name]
    
    def calculate(self, *args, **kwargs) -> float:
        """
        Calculate the total reward.
        
        Args:
            *args: Positional arguments to pass to component functions
            **kwargs: Keyword arguments to pass to component functions
            
        Returns:
            Total reward value
        """
        total_reward = 0.0
        
        for component in self.components:
            reward_value = component.calculate(*args, **kwargs)
            total_reward += reward_value
        
        return total_reward
    
    def get_component_values(self, *args, **kwargs) -> dict:
        """
        Get the values of all components.
        
        Args:
            *args: Positional arguments to pass to component functions
            **kwargs: Keyword arguments to pass to component functions
            
        Returns:
            Dictionary mapping component names to values
        """
        return {
            component.name: component.calculate(*args, **kwargs) / component.weight
            for component in self.components
            if component.enabled
        }
    
    def get_component_history(self) -> dict:
        """
        Get the history of all components.
        
        Returns:
            Dictionary mapping component names to history lists
        """
        return {
            component.name: component.history
            for component in self.components
        }