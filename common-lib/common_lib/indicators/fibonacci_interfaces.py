"""
Fibonacci Indicator Interfaces Module

This module provides interfaces for Fibonacci-based technical analysis components,
helping to break circular dependencies between services and tests.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd


class TrendDirectionType(str, Enum):
    """Enum for trend direction types."""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class IFibonacciBase(ABC):
    """Base interface for all Fibonacci indicators."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the indicator."""
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
    
    @staticmethod
    @abstractmethod
    def get_info() -> Dict[str, Any]:
        """
        Get information about the indicator.
        
        Returns:
            Dictionary with indicator information
        """
        pass


class IFibonacciRetracement(IFibonacciBase):
    """Interface for Fibonacci Retracement indicator."""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with retracement levels added
        """
        pass


class IFibonacciExtension(IFibonacciBase):
    """Interface for Fibonacci Extension indicator."""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci extension levels.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with extension levels added
        """
        pass


class IFibonacciFan(IFibonacciBase):
    """Interface for Fibonacci Fan indicator."""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci fan levels.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with fan levels added
        """
        pass


class IFibonacciTimeZones(IFibonacciBase):
    """Interface for Fibonacci Time Zones indicator."""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci time zones.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with time zone levels added
        """
        pass


class IFibonacciCircles(IFibonacciBase):
    """Interface for Fibonacci Circles indicator."""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci circle levels.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with circle levels added
        """
        pass


class IFibonacciClusters(IFibonacciBase):
    """Interface for Fibonacci Clusters indicator."""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci cluster levels.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with cluster levels added
        """
        pass


class IFibonacciUtils(ABC):
    """Interface for Fibonacci utility functions."""
    
    @staticmethod
    @abstractmethod
    def generate_fibonacci_sequence(n: int) -> List[int]:
        """
        Generate a Fibonacci sequence of length n.
        
        Args:
            n: Length of the sequence
            
        Returns:
            List of Fibonacci numbers
        """
        pass
    
    @staticmethod
    @abstractmethod
    def fibonacci_ratios() -> List[float]:
        """
        Get common Fibonacci ratios.
        
        Returns:
            List of Fibonacci ratios
        """
        pass
    
    @staticmethod
    @abstractmethod
    def calculate_fibonacci_retracement_levels(
        start_price: float,
        end_price: float,
        levels: List[float] = None
    ) -> Dict[float, float]:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            start_price: Starting price
            end_price: Ending price
            levels: Optional list of Fibonacci levels
            
        Returns:
            Dictionary mapping levels to prices
        """
        pass
    
    @staticmethod
    @abstractmethod
    def calculate_fibonacci_extension_levels(
        start_price: float,
        end_price: float,
        retracement_price: float,
        levels: List[float] = None
    ) -> Dict[float, float]:
        """
        Calculate Fibonacci extension levels.
        
        Args:
            start_price: Starting price
            end_price: Ending price
            retracement_price: Retracement price
            levels: Optional list of Fibonacci levels
            
        Returns:
            Dictionary mapping levels to prices
        """
        pass
