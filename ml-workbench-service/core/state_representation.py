"""
State Representation for RL Environments

This module provides utilities for managing state representation.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from .feature_extractors import (
    FeatureExtractor,
    MarketDataFeatureExtractor,
    TechnicalIndicatorExtractor,
    BrokerStateExtractor,
    OrderBookExtractor,
    NewsSentimentExtractor
)


class StateRepresentation:
    """
    Manages the state representation for RL environments.
    
    This class combines multiple feature extractors to create a complete
    state representation for the RL agent.
    """
    
    def __init__(self):
        """Initialize the state representation."""
        self.feature_extractors = {}
        self.feature_dimensions = {}
        self.total_dimension = 0
    
    def add_extractor(self, name: str, extractor: FeatureExtractor, dimension: int):
        """
        Add a feature extractor.
        
        Args:
            name: Name of the extractor
            extractor: Feature extractor instance
            dimension: Dimension of the extracted features
        """
        self.feature_extractors[name] = extractor
        self.feature_dimensions[name] = dimension
        self.total_dimension += dimension
    
    def get_state(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Get the complete state representation.
        
        Args:
            data: Dictionary of data for different extractors
            
        Returns:
            Complete state representation as a numpy array
        """
        state = []
        
        for name, extractor in self.feature_extractors.items():
            if name in data:
                features = extractor.extract(data[name])
                state.extend(features)
            else:
                # Data not available, pad with zeros
                state.extend([0] * self.feature_dimensions[name])
        
        # Ensure the state has the correct dimension
        if len(state) < self.total_dimension:
            state.extend([0] * (self.total_dimension - len(state)))
        elif len(state) > self.total_dimension:
            state = state[:self.total_dimension]
        
        return np.array(state, dtype=np.float32)
    
    def get_feature_slice(self, state: np.ndarray, feature_name: str) -> np.ndarray:
        """
        Get a slice of the state corresponding to a specific feature.
        
        Args:
            state: Complete state representation
            feature_name: Name of the feature to extract
            
        Returns:
            Slice of the state for the specified feature
        """
        if feature_name not in self.feature_dimensions:
            return np.array([])
        
        # Calculate the start and end indices for the feature
        start_idx = 0
        for name, dimension in self.feature_dimensions.items():
            if name == feature_name:
                end_idx = start_idx + dimension
                return state[start_idx:end_idx]
            start_idx += dimension
        
        return np.array([])


def create_forex_state_representation(
    timeframes: List[str],
    features: List[str],
    lookback_periods: int,
    include_technical_indicators: bool = True,
    include_broker_state: bool = True,
    include_order_book: bool = True,
    include_news_sentiment: bool = False,
    news_sentiment_dimension: int = 0
) -> Tuple[StateRepresentation, int]:
    """
    Create a state representation for forex trading.
    
    Args:
        timeframes: List of timeframes
        features: List of features
        lookback_periods: Number of past periods to include
        include_technical_indicators: Whether to include technical indicators
        include_broker_state: Whether to include broker state
        include_order_book: Whether to include order book
        include_news_sentiment: Whether to include news sentiment
        news_sentiment_dimension: Dimension of news sentiment features
        
    Returns:
        Tuple of (state_representation, total_dimension)
    """
    state_rep = StateRepresentation()
    
    # Add market data extractor
    market_data_dim = len(timeframes) * len(features) * lookback_periods
    state_rep.add_extractor(
        'market_data',
        MarketDataFeatureExtractor(features, lookback_periods),
        market_data_dim
    )
    
    # Add technical indicator extractor if enabled
    if include_technical_indicators:
        indicators = ['sma5', 'sma20', 'ema5', 'ema20', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_lower']
        indicator_dim = len(timeframes) * len(indicators)
        state_rep.add_extractor(
            'technical_indicators',
            TechnicalIndicatorExtractor(indicators),
            indicator_dim
        )
    
    # Add broker state extractor if enabled
    if include_broker_state:
        broker_state_dim = 5  # account_balance, current_position, current_pnl, total_trades, profitable_trades
        state_rep.add_extractor(
            'broker_state',
            BrokerStateExtractor(),
            broker_state_dim
        )
    
    # Add order book extractor if enabled
    if include_order_book:
        order_book_dim = 5 * 4  # 5 levels, 2 values per level (price, volume), 2 sides (bid, ask)
        state_rep.add_extractor(
            'order_book',
            OrderBookExtractor(levels=5),
            order_book_dim
        )
    
    # Add news sentiment extractor if enabled
    if include_news_sentiment and news_sentiment_dimension > 0:
        state_rep.add_extractor(
            'news_sentiment',
            NewsSentimentExtractor(news_sentiment_dimension),
            news_sentiment_dimension
        )
    
    return state_rep, state_rep.total_dimension