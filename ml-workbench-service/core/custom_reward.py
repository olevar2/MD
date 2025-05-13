"""
Custom Reward Components

This module provides support for custom reward functions.
"""

from typing import Callable, Dict, Any

from .base_reward import RewardComponent, CompositeReward


class CustomReward(CompositeReward):
    """
    A reward function that uses a custom function.
    
    This reward function allows for complete customization of the reward calculation,
    while still maintaining the component-based structure for analysis.
    """
    
    def __init__(self, custom_function: Callable):
        """
        Initialize the custom reward function.
        
        Args:
            custom_function: Custom reward calculation function
        """
        super().__init__()
        
        # Add custom function component
        self.add_component(
            RewardComponent(
                name="custom",
                weight=1.0,
                function=custom_function,
                description="Custom reward function"
            )
        )


def create_custom_reward(custom_function: Callable) -> CompositeReward:
    """
    Create a custom reward function.
    
    Args:
        custom_function: Custom reward calculation function
        
    Returns:
        Composite reward function
    """
    reward = CompositeReward()
    
    # Add custom function component
    reward.add_component(
        RewardComponent(
            name="custom",
            weight=1.0,
            function=custom_function,
            description="Custom reward function"
        )
    )
    
    return reward


class NewsAlignmentReward(RewardComponent):
    """
    A reward component for alignment with news sentiment.
    
    This component rewards the agent for taking positions that align with
    the sentiment of recent news events.
    """
    
    def __init__(self, weight: float = 0.3):
        """
        Initialize the news alignment reward component.
        
        Args:
            weight: Weight for this component
        """
        super().__init__(
            name="news_alignment_bonus",
            weight=weight,
            function=self._calculate_news_alignment_reward,
            description="Bonus for trading aligned with significant news events"
        )
    
    def _calculate_news_alignment_reward(self, env) -> float:
        """
        Calculate reward for alignment with significant news events.
        
        Args:
            env: Environment instance
            
        Returns:
            Reward value
        """
        if not hasattr(env, 'news_sentiment_simulator') or not env.news_sentiment_simulator or not env.current_position:
            return 0.0
        
        # Get very recent high-impact news
        recent_events = env.news_sentiment_simulator.get_recent_events(
            env.current_timestamp,
            lookback_hours=2,
            min_impact="HIGH",
            relevant_currencies=env.symbol.replace('/', ',')
        )
        
        if not recent_events:
            return 0.0
        
        # Calculate average sentiment for these events
        avg_sentiment = sum(event.sentiment_score for event in recent_events) / len(recent_events)
        
        # If position aligns with sentiment direction, give reward
        position_sign = 1 if env.current_position > 0 else (-1 if env.current_position < 0 else 0)
        sentiment_sign = 1 if avg_sentiment > 0 else (-1 if avg_sentiment < 0 else 0)
        
        alignment_score = position_sign * sentiment_sign
        
        # Scale by the magnitude of sentiment and position
        scaled_score = alignment_score * abs(avg_sentiment) * min(1.0, abs(env.current_position))
        
        return scaled_score