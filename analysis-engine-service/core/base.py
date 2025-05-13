"""
Base classes for Advanced Technical Analysis

This module provides the foundation classes for all advanced technical analysis modules
including base classes for analytical methods, pattern recognition, and common data structures.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import uuid


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class ConfidenceLevel(Enum):
    """Confidence levels for technical analysis results"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5


class MarketDirection(Enum):
    """Market direction indicators"""
    BULLISH = auto()
    BEARISH = auto()
    NEUTRAL = auto()
    CONSOLIDATION = auto()
    VOLATILE = auto()


class AnalysisTimeframe(Enum):
    """Standard timeframes for analysis"""
    M1 = '1m'
    M5 = '5m'
    M15 = '15m'
    M30 = '30m'
    H1 = '1h'
    H4 = '4h'
    D1 = '1d'
    W1 = '1w'
    MN1 = '1M'


@dataclass
class PatternResult:
    """Standard result format for pattern detection"""
    pattern_id: str = field(default_factory=lambda : str(uuid.uuid4()))
    pattern_name: str = ''
    timeframe: AnalysisTimeframe = AnalysisTimeframe.D1
    direction: MarketDirection = MarketDirection.NEUTRAL
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    start_price: Optional[float] = None
    end_price: Optional[float] = None
    target_prices: List[float] = field(default_factory=list)
    stop_loss: Optional[float] = None
    success_probability: float = 0.0
    notes: str = ''
    pattern_points: Dict[str, Tuple[datetime, float]] = field(default_factory
        =dict)
    effectiveness_score: float = 0.0

    def to_dict(self) ->Dict[str, Any]:
        """Convert the dataclass instance to a dictionary."""
        return {'pattern_id': self.pattern_id, 'pattern_name': self.
            pattern_name, 'timeframe': self.timeframe.value, 'direction':
            self.direction.name, 'confidence': self.confidence.name,
            'start_time': self.start_time.isoformat() if self.start_time else
            None, 'end_time': self.end_time.isoformat() if self.end_time else
            None, 'start_price': self.start_price, 'end_price': self.
            end_price, 'target_prices': self.target_prices, 'stop_loss':
            self.stop_loss, 'success_probability': self.success_probability,
            'notes': self.notes, 'pattern_points': {k: (v[0].isoformat(), v
            [1]) for k, v in self.pattern_points.items()},
            'effectiveness_score': self.effectiveness_score}


class AdvancedAnalysisBase(ABC):
    """
    Base class for all advanced technical analysis techniques
    
    This class provides the foundation for both standard and incremental calculations
    of advanced technical analysis indicators and metrics.
    """

    def __init__(self, name: str, parameters: Dict[str, Any]=None):
        """
        Initialize the advanced analysis calculator
        
        Args:
            name: Name of the analysis technique
            parameters: Dictionary of parameters for the technique
        """
        self.name = name
        self.parameters = parameters or {}
        self.is_incremental = False

    @abstractmethod
    def calculate(self, df: pd.DataFrame) ->pd.DataFrame:
        """
        Calculate the analysis on a full DataFrame
        
        Args:
            df: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with calculated values
        """
        pass

    def initialize_incremental(self) ->Dict[str, Any]:
        """
        Initialize state for incremental calculation
        
        Returns:
            State dictionary for incremental updates
        """
        raise NotImplementedError(
            f'Incremental calculation not implemented for {self.name}')

    @with_resilience('update_incremental')
    def update_incremental(self, state: Dict[str, Any], new_data: Dict[str,
        float]) ->Dict[str, Any]:
        """
        Update calculation with new data incrementally
        
        Args:
            state: Current state dictionary
            new_data: New data point
            
        Returns:
            Updated state and results
        """
        raise NotImplementedError(
            f'Incremental calculation not implemented for {self.name}')

    def log_effectiveness(self, result: Any, outcome: Dict[str, Any]) ->None:
        """
        Log the effectiveness of this analysis technique
        
        Args:
            result: The original analysis result
            outcome: The actual market outcome
        """
        pass


class PatternRecognitionBase(AdvancedAnalysisBase):
    """
    Base class for pattern recognition algorithms
    
    This class extends AdvancedAnalysisBase with pattern-specific
    functionality including detection and confidence scoring.
    """

    def __init__(self, name: str, parameters: Dict[str, Any]=None):
        """
        Initialize the pattern recognition calculator
        
        Args:
            name: Name of the pattern recognition technique
            parameters: Dictionary of parameters for the technique
        """
        super().__init__(name, parameters)
        self.min_bars_required = self.parameters.get('min_bars_required', 10)

    @abstractmethod
    def find_patterns(self, df: pd.DataFrame) ->List[PatternResult]:
        """
        Find all instances of the pattern in the DataFrame
        
        Args:
            df: DataFrame containing OHLCV data
            
        Returns:
            List of PatternResult objects
        """
        pass

    @with_analysis_resilience('calculate_confidence')
    def calculate_confidence(self, pattern: dict) ->ConfidenceLevel:
        """
        Calculate the confidence level of a detected pattern
        
        Args:
            pattern: Dictionary containing pattern details
            
        Returns:
            ConfidenceLevel enum value
        """
        return ConfidenceLevel.MEDIUM

    def calculate(self, df: pd.DataFrame) ->pd.DataFrame:
        """
        Identify patterns and add to DataFrame
        
        Args:
            df: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with pattern information
        """
        patterns = self.find_patterns(df)
        result_df = df.copy()
        pattern_col = f'{self.name}_pattern'
        result_df[pattern_col] = False
        for pattern in patterns:
            if pattern.start_time and pattern.end_time:
                mask = (result_df.index >= pattern.start_time) & (result_df
                    .index <= pattern.end_time)
                result_df.loc[mask, pattern_col] = True
        return result_df

    def initialize_incremental(self) ->Dict[str, Any]:
        """
        Initialize state for incremental pattern detection
        
        Returns:
            State dictionary for incremental updates
        """
        return {'buffer': [], 'potential_patterns': [], 'complete_patterns':
            [], 'last_update': None}

    @with_resilience('update_incremental')
    def update_incremental(self, state: Dict[str, Any], new_data: Dict[str,
        Any]) ->Dict[str, Any]:
        """
        Update pattern detection with new data incrementally
        
        Args:
            state: Current state dictionary
            new_data: New data point
            
        Returns:
            Updated state and any newly detected patterns
        """
        state['buffer'].append(new_data)
        state['last_update'] = datetime.now()
        max_buffer_size = self.parameters.get('max_buffer_size', 200)
        if len(state['buffer']) > max_buffer_size:
            state['buffer'] = state['buffer'][-max_buffer_size:]
        return state


def normalize_price_series(prices: np.ndarray) ->np.ndarray:
    """
    Normalize price series to 0-1 range for pattern comparison
    
    Args:
        prices: Array of price values
    
    Returns:
        Normalized price array
    """
    min_price = np.min(prices)
    price_range = np.max(prices) - min_price
    if price_range == 0:
        return np.zeros_like(prices)
    return (prices - min_price) / price_range


def detect_swings(df: pd.DataFrame, lookback: int=5, price_col: str='close'
    ) ->pd.DataFrame:
    """
    Detect swing highs and lows in price series
    
    Args:
        df: DataFrame with price data
        lookback: Number of bars to look back/forward
        price_col: Column name for price data
        
    Returns:
        DataFrame with swing_high and swing_low columns
    """
    df_copy = df.copy()
    df_copy['swing_high'] = False
    df_copy['swing_low'] = False
    for i in range(lookback, len(df_copy) - lookback):
        if all(df_copy.iloc[i][price_col] > df_copy.iloc[i - j][price_col] for
            j in range(1, lookback + 1)) and all(df_copy.iloc[i][price_col] >
            df_copy.iloc[i + j][price_col] for j in range(1, lookback + 1)):
            df_copy.loc[df_copy.index[i], 'swing_high'] = True
        if all(df_copy.iloc[i][price_col] < df_copy.iloc[i - j][price_col] for
            j in range(1, lookback + 1)) and all(df_copy.iloc[i][price_col] <
            df_copy.iloc[i + j][price_col] for j in range(1, lookback + 1)):
            df_copy.loc[df_copy.index[i], 'swing_low'] = True
    return df_copy


def calculate_retracement_levels(start_price: float, end_price: float,
    levels: List[float]=None) ->Dict[str, float]:
    """
    Calculate retracement levels (e.g., Fibonacci) between two price points
    
    Args:
        start_price: Starting price point
        end_price: Ending price point
        levels: Specific retracement levels (defaults to Fibonacci)
        
    Returns:
        Dictionary of retracement levels and prices
    """
    if levels is None:
        levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    price_range = end_price - start_price
    result = {}
    for level in levels:
        level_price = start_price + price_range * level
        result[f'{level:.3f}'] = level_price
    return result


def calculate_projection_levels(start_price: float, end_price: float,
    levels: List[float]=None) ->Dict[str, float]:
    """
    Calculate projection levels (e.g., Fibonacci) from a price move
    
    Args:
        start_price: Starting price point
        end_price: Ending price point
        levels: Specific projection levels (defaults to Fibonacci)
        
    Returns:
        Dictionary of projection levels and prices
    """
    if levels is None:
        levels = [1.0, 1.272, 1.414, 1.618, 2.0, 2.618]
    price_range = end_price - start_price
    result = {}
    for level in levels:
        level_price = start_price + price_range * level
        result[f'{level:.3f}'] = level_price
    return result
