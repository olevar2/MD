"""
Fibonacci Adapter Module

This module provides adapter implementations for Fibonacci indicator interfaces,
helping to break circular dependencies between feature-store-service and tests.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from enum import Enum

from common_lib.indicators.fibonacci_interfaces import (
    TrendDirectionType,
    IFibonacciBase,
    IFibonacciRetracement,
    IFibonacciExtension,
    IFibonacciFan,
    IFibonacciTimeZones,
    IFibonacciCircles,
    IFibonacciClusters,
    IFibonacciUtils
)

# Import the facade components
from core.facade_2 import (
    TrendDirection,
    FibonacciBase,
    FibonacciRetracement,
    FibonacciExtension,
    FibonacciFan,
    FibonacciTimeZones,
    FibonacciCircles,
    FibonacciClusters,
    generate_fibonacci_sequence,
    fibonacci_ratios,
    calculate_fibonacci_retracement_levels,
    calculate_fibonacci_extension_levels,
    format_fibonacci_level,
    is_golden_ratio,
    is_fibonacci_ratio
)

# Map TrendDirection to TrendDirectionType
TREND_DIRECTION_MAP = {
    TrendDirection.UPTREND: TrendDirectionType.UPTREND,
    TrendDirection.DOWNTREND: TrendDirectionType.DOWNTREND
}


class FibonacciBaseAdapter(IFibonacciBase):
    """Adapter for FibonacciBase."""

    def __init__(self, fibonacci_base: FibonacciBase):
        """
        Initialize the adapter.

        Args:
            fibonacci_base: FibonacciBase instance to adapt
        """
        self._fibonacci_base = fibonacci_base

    @property
    def name(self) -> str:
        """Get the name of the indicator."""
        return self._fibonacci_base.name

    @property
    def params(self) -> Dict[str, Any]:
        """Get the parameters for the indicator."""
        return self._fibonacci_base.params

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the indicator values.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with indicator values added as columns
        """
        return self._fibonacci_base.calculate(data)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        """
        Get information about the indicator.

        Returns:
            Dictionary with indicator information
        """
        return FibonacciBase.get_info()


class FibonacciRetracementAdapter(FibonacciBaseAdapter, IFibonacciRetracement):
    """Adapter for FibonacciRetracement."""

    def __init__(self, fibonacci_retracement: FibonacciRetracement = None, **kwargs):
        """
        Initialize the adapter.

        Args:
            fibonacci_retracement: FibonacciRetracement instance to adapt
            **kwargs: Parameters to pass to FibonacciRetracement constructor if no instance is provided
        """
        if fibonacci_retracement is None:
            fibonacci_retracement = FibonacciRetracement(**kwargs)
        super().__init__(fibonacci_retracement)
        self._fibonacci_retracement = fibonacci_retracement

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci retracement levels.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with retracement levels added
        """
        return self._fibonacci_retracement.calculate(data)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        """
        Get information about the indicator.

        Returns:
            Dictionary with indicator information
        """
        return FibonacciRetracement.get_info()


class FibonacciExtensionAdapter(FibonacciBaseAdapter, IFibonacciExtension):
    """Adapter for FibonacciExtension."""

    def __init__(self, fibonacci_extension: FibonacciExtension = None, **kwargs):
        """
        Initialize the adapter.

        Args:
            fibonacci_extension: FibonacciExtension instance to adapt
            **kwargs: Parameters to pass to FibonacciExtension constructor if no instance is provided
        """
        if fibonacci_extension is None:
            fibonacci_extension = FibonacciExtension(**kwargs)
        super().__init__(fibonacci_extension)
        self._fibonacci_extension = fibonacci_extension

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci extension levels.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with extension levels added
        """
        return self._fibonacci_extension.calculate(data)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        """
        Get information about the indicator.

        Returns:
            Dictionary with indicator information
        """
        return FibonacciExtension.get_info()


class FibonacciFanAdapter(FibonacciBaseAdapter, IFibonacciFan):
    """Adapter for FibonacciFan."""

    def __init__(self, fibonacci_fan: FibonacciFan = None, **kwargs):
        """
        Initialize the adapter.

        Args:
            fibonacci_fan: FibonacciFan instance to adapt
            **kwargs: Parameters to pass to FibonacciFan constructor if no instance is provided
        """
        if fibonacci_fan is None:
            fibonacci_fan = FibonacciFan(**kwargs)
        super().__init__(fibonacci_fan)
        self._fibonacci_fan = fibonacci_fan

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci fan levels.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with fan levels added
        """
        return self._fibonacci_fan.calculate(data)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        """
        Get information about the indicator.

        Returns:
            Dictionary with indicator information
        """
        return FibonacciFan.get_info()


class FibonacciTimeZonesAdapter(FibonacciBaseAdapter, IFibonacciTimeZones):
    """Adapter for FibonacciTimeZones."""

    def __init__(self, fibonacci_time_zones: FibonacciTimeZones = None, **kwargs):
        """
        Initialize the adapter.

        Args:
            fibonacci_time_zones: FibonacciTimeZones instance to adapt
            **kwargs: Parameters to pass to FibonacciTimeZones constructor if no instance is provided
        """
        if fibonacci_time_zones is None:
            fibonacci_time_zones = FibonacciTimeZones(**kwargs)
        super().__init__(fibonacci_time_zones)
        self._fibonacci_time_zones = fibonacci_time_zones

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci time zones.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with time zone levels added
        """
        return self._fibonacci_time_zones.calculate(data)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        """
        Get information about the indicator.

        Returns:
            Dictionary with indicator information
        """
        return FibonacciTimeZones.get_info()


class FibonacciCirclesAdapter(FibonacciBaseAdapter, IFibonacciCircles):
    """Adapter for FibonacciCircles."""

    def __init__(self, fibonacci_circles: FibonacciCircles = None, **kwargs):
        """
        Initialize the adapter.

        Args:
            fibonacci_circles: FibonacciCircles instance to adapt
            **kwargs: Parameters to pass to FibonacciCircles constructor if no instance is provided
        """
        if fibonacci_circles is None:
            fibonacci_circles = FibonacciCircles(**kwargs)
        super().__init__(fibonacci_circles)
        self._fibonacci_circles = fibonacci_circles

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci circle levels.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with circle levels added
        """
        return self._fibonacci_circles.calculate(data)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        """
        Get information about the indicator.

        Returns:
            Dictionary with indicator information
        """
        return FibonacciCircles.get_info()


class FibonacciClustersAdapter(FibonacciBaseAdapter, IFibonacciClusters):
    """Adapter for FibonacciClusters."""

    def __init__(self, fibonacci_clusters: FibonacciClusters = None, **kwargs):
        """
        Initialize the adapter.

        Args:
            fibonacci_clusters: FibonacciClusters instance to adapt
            **kwargs: Parameters to pass to FibonacciClusters constructor if no instance is provided
        """
        if fibonacci_clusters is None:
            fibonacci_clusters = FibonacciClusters(**kwargs)
        super().__init__(fibonacci_clusters)
        self._fibonacci_clusters = fibonacci_clusters

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci cluster levels.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with cluster levels added
        """
        return self._fibonacci_clusters.calculate(data)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        """
        Get information about the indicator.

        Returns:
            Dictionary with indicator information
        """
        return FibonacciClusters.get_info()


class FibonacciUtilsAdapter(IFibonacciUtils):
    """Adapter for Fibonacci utility functions."""

    @staticmethod
    def generate_fibonacci_sequence(n: int) -> List[int]:
        """
        Generate a Fibonacci sequence of length n.

        Args:
            n: Length of the sequence

        Returns:
            List of Fibonacci numbers
        """
        return generate_fibonacci_sequence(n)

    @staticmethod
    def fibonacci_ratios() -> List[float]:
        """
        Get common Fibonacci ratios.

        Returns:
            List of Fibonacci ratios
        """
        return fibonacci_ratios()

    @staticmethod
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
        return calculate_fibonacci_retracement_levels(start_price, end_price, levels)

    @staticmethod
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
        return calculate_fibonacci_extension_levels(start_price, end_price, retracement_price, levels)
