"""
Indicator Interfaces Module

This module provides interfaces for technical indicators and advanced analysis components,
helping to break circular dependencies between services.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd


class IndicatorCategory(str, Enum):
    """Categories of technical indicators."""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    PATTERN = "pattern"
    OSCILLATOR = "oscillator"
    CUSTOM = "custom"


class IBaseIndicator(ABC):
    """Interface for base indicator functionality."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the indicator."""
        pass
    
    @property
    @abstractmethod
    def category(self) -> IndicatorCategory:
        """Get the category of the indicator."""
        pass
    
    @property
    @abstractmethod
    def params(self) -> Dict[str, Any]:
        """Get the parameters for the indicator."""
        pass
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the indicator values.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicator values added as columns
        """
        pass
    
    @abstractmethod
    def get_column_names(self) -> List[str]:
        """
        Get the names of columns added by this indicator.
        
        Returns:
            List of column names
        """
        pass


class IAdvancedIndicator(IBaseIndicator):
    """Interface for advanced indicators with additional functionality."""
    
    @abstractmethod
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get trading signals from the indicator.
        
        Args:
            data: DataFrame with OHLCV data and indicator values
            
        Returns:
            DataFrame with signal columns added
        """
        pass
    
    @abstractmethod
    def get_visualization_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get data for visualization.
        
        Args:
            data: DataFrame with OHLCV data and indicator values
            
        Returns:
            Dictionary with visualization data
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the indicator.
        
        Returns:
            Dictionary with indicator metadata
        """
        pass


class IPatternRecognizer(ABC):
    """Interface for pattern recognition components."""
    
    @abstractmethod
    def find_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Find patterns in the data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with pattern columns added
        """
        pass
    
    @abstractmethod
    def get_pattern_info(self, pattern_id: str) -> Dict[str, Any]:
        """
        Get information about a specific pattern.
        
        Args:
            pattern_id: ID of the pattern
            
        Returns:
            Dictionary with pattern information
        """
        pass
    
    @abstractmethod
    def get_supported_patterns(self) -> List[str]:
        """
        Get a list of supported patterns.
        
        Returns:
            List of pattern names
        """
        pass


class IFibonacciAnalyzer(ABC):
    """Interface for Fibonacci analysis components."""
    
    @abstractmethod
    def calculate_retracements(
        self,
        data: pd.DataFrame,
        high_col: str = 'high',
        low_col: str = 'low',
        levels: List[float] = None
    ) -> pd.DataFrame:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            data: DataFrame with OHLCV data
            high_col: Name of the high price column
            low_col: Name of the low price column
            levels: Optional list of Fibonacci levels
            
        Returns:
            DataFrame with retracement levels added
        """
        pass
    
    @abstractmethod
    def calculate_extensions(
        self,
        data: pd.DataFrame,
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        levels: List[float] = None
    ) -> pd.DataFrame:
        """
        Calculate Fibonacci extension levels.
        
        Args:
            data: DataFrame with OHLCV data
            high_col: Name of the high price column
            low_col: Name of the low price column
            close_col: Name of the close price column
            levels: Optional list of Fibonacci levels
            
        Returns:
            DataFrame with extension levels added
        """
        pass
    
    @abstractmethod
    def calculate_arcs(
        self,
        data: pd.DataFrame,
        high_col: str = 'high',
        low_col: str = 'low',
        levels: List[float] = None
    ) -> pd.DataFrame:
        """
        Calculate Fibonacci arcs.
        
        Args:
            data: DataFrame with OHLCV data
            high_col: Name of the high price column
            low_col: Name of the low price column
            levels: Optional list of Fibonacci levels
            
        Returns:
            DataFrame with arc levels added
        """
        pass
    
    @abstractmethod
    def calculate_fans(
        self,
        data: pd.DataFrame,
        high_col: str = 'high',
        low_col: str = 'low',
        levels: List[float] = None
    ) -> pd.DataFrame:
        """
        Calculate Fibonacci fans.
        
        Args:
            data: DataFrame with OHLCV data
            high_col: Name of the high price column
            low_col: Name of the low price column
            levels: Optional list of Fibonacci levels
            
        Returns:
            DataFrame with fan levels added
        """
        pass
    
    @abstractmethod
    def calculate_time_zones(
        self,
        data: pd.DataFrame,
        pivot_idx: int = None,
        levels: List[int] = None
    ) -> pd.DataFrame:
        """
        Calculate Fibonacci time zones.
        
        Args:
            data: DataFrame with OHLCV data
            pivot_idx: Optional pivot index
            levels: Optional list of Fibonacci levels
            
        Returns:
            DataFrame with time zone levels added
        """
        pass


class IIndicatorRegistry(ABC):
    """Interface for indicator registry components."""
    
    @abstractmethod
    def register_indicator(self, name: str, indicator_class: Any) -> None:
        """
        Register an indicator.
        
        Args:
            name: Name of the indicator
            indicator_class: Indicator class or instance
        """
        pass
    
    @abstractmethod
    def get_indicator(self, name: str) -> Any:
        """
        Get an indicator by name.
        
        Args:
            name: Name of the indicator
            
        Returns:
            Indicator class or instance
        """
        pass
    
    @abstractmethod
    def get_all_indicators(self) -> Dict[str, Any]:
        """
        Get all registered indicators.
        
        Returns:
            Dictionary mapping indicator names to classes or instances
        """
        pass
    
    @abstractmethod
    def get_indicators_by_category(self, category: str) -> Dict[str, Any]:
        """
        Get indicators by category.
        
        Args:
            category: Category name
            
        Returns:
            Dictionary mapping indicator names to classes or instances
        """
        pass


class IIndicatorAdapter(ABC):
    """Interface for indicator adapters."""
    
    @abstractmethod
    def adapt(self, source_indicator: Any) -> IBaseIndicator:
        """
        Adapt a source indicator to the IBaseIndicator interface.
        
        Args:
            source_indicator: Source indicator instance
            
        Returns:
            Adapted indicator implementing IBaseIndicator
        """
        pass
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicator values using the adapted indicator.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicator values added
        """
        pass