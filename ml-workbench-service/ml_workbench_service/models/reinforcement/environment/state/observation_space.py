"""
Observation Space Builder

This module provides utilities for building observation spaces for RL environments.
"""

import numpy as np
from gym import spaces
from typing import List, Dict, Any, Optional, Tuple


class ObservationSpaceBuilder:
    """
    Builder for constructing observation spaces.
    
    This class helps construct complex observation spaces by adding different
    types of features and calculating the total dimension.
    """
    
    def __init__(self):
        """Initialize the observation space builder."""
        self.dimensions = 0
        self.feature_dimensions = {}
    
    def add_market_data_features(self, 
                               timeframes: List[str], 
                               features: List[str], 
                               lookback_periods: int) -> 'ObservationSpaceBuilder':
        """
        Add market data features to the observation space.
        
        Args:
            timeframes: List of timeframes
            features: List of features (e.g., 'open', 'high', 'low', 'close')
            lookback_periods: Number of past periods to include
            
        Returns:
            Self for method chaining
        """
        feature_dim = len(timeframes) * len(features) * lookback_periods
        self.dimensions += feature_dim
        self.feature_dimensions['market_data'] = feature_dim
        return self
    
    def add_technical_indicators(self, 
                               timeframes: List[str], 
                               indicators: List[str]) -> 'ObservationSpaceBuilder':
        """
        Add technical indicator features to the observation space.
        
        Args:
            timeframes: List of timeframes
            indicators: List of indicators
            
        Returns:
            Self for method chaining
        """
        feature_dim = len(timeframes) * len(indicators)
        self.dimensions += feature_dim
        self.feature_dimensions['technical_indicators'] = feature_dim
        return self
    
    def add_broker_state(self, num_features: int = 5) -> 'ObservationSpaceBuilder':
        """
        Add broker state features to the observation space.
        
        Args:
            num_features: Number of broker state features
            
        Returns:
            Self for method chaining
        """
        self.dimensions += num_features
        self.feature_dimensions['broker_state'] = num_features
        return self
    
    def add_order_book(self, levels: int = 5) -> 'ObservationSpaceBuilder':
        """
        Add order book features to the observation space.
        
        Args:
            levels: Number of order book levels
            
        Returns:
            Self for method chaining
        """
        feature_dim = levels * 4  # price and volume for both bid and ask
        self.dimensions += feature_dim
        self.feature_dimensions['order_book'] = feature_dim
        return self
    
    def add_news_sentiment(self, num_features: int) -> 'ObservationSpaceBuilder':
        """
        Add news sentiment features to the observation space.
        
        Args:
            num_features: Number of news sentiment features
            
        Returns:
            Self for method chaining
        """
        self.dimensions += num_features
        self.feature_dimensions['news_sentiment'] = num_features
        return self
    
    def add_custom_features(self, name: str, num_features: int) -> 'ObservationSpaceBuilder':
        """
        Add custom features to the observation space.
        
        Args:
            name: Name of the feature group
            num_features: Number of features
            
        Returns:
            Self for method chaining
        """
        self.dimensions += num_features
        self.feature_dimensions[name] = num_features
        return self
    
    def build(self) -> Tuple[spaces.Box, Dict[str, int]]:
        """
        Build the observation space.
        
        Returns:
            Tuple of (observation_space, feature_dimensions)
        """
        observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.dimensions,), 
            dtype=np.float32
        )
        
        return observation_space, self.feature_dimensions