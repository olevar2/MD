"""
Fibonacci Base Module

This module provides base classes and utilities for Fibonacci analysis.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass

from core_foundations.utils.logger import get_logger
from core_foundations.exceptions.analysis_exceptions import FibonacciAnalysisError

logger = get_logger(__name__)


class FibonacciDirection(str, Enum):
    """Direction of Fibonacci analysis."""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"


class FibonacciType(str, Enum):
    """Type of Fibonacci analysis."""
    RETRACEMENT = "retracement"
    EXTENSION = "extension"
    ARCS = "arcs"
    FANS = "fans"
    TIME_ZONES = "time_zones"


@dataclass
class FibonacciLevel:
    """Represents a Fibonacci level."""
    ratio: float
    value: float
    name: str
    is_key_level: bool = False
    description: Optional[str] = None


@dataclass
class FibonacciPoint:
    """Represents a point in Fibonacci analysis."""
    index: int
    price: float
    timestamp: str


class FibonacciBase:
    """Base class for Fibonacci analysis."""
    
    # Standard Fibonacci ratios
    STANDARD_RATIOS = {
        0.0: "0%",
        0.236: "23.6%",
        0.382: "38.2%",
        0.5: "50%",
        0.618: "61.8%",
        0.786: "78.6%",
        1.0: "100%",
        1.272: "127.2%",
        1.414: "141.4%",
        1.618: "161.8%",
        2.0: "200%",
        2.618: "261.8%",
        3.618: "361.8%",
        4.236: "423.6%"
    }
    
    # Key levels that are most important
    KEY_LEVELS = [0.0, 0.382, 0.618, 1.0, 1.618, 2.618]
    
    def __init__(
        self,
        fibonacci_type: FibonacciType,
        ratios: Optional[Dict[float, str]] = None,
        key_levels: Optional[List[float]] = None
    ):
        """
        Initialize the Fibonacci base.
        
        Args:
            fibonacci_type: Type of Fibonacci analysis
            ratios: Custom Fibonacci ratios (optional)
            key_levels: Custom key levels (optional)
        """
        self.fibonacci_type = fibonacci_type
        self.ratios = ratios or self.STANDARD_RATIOS
        self.key_levels = key_levels or self.KEY_LEVELS
        
        # Sort ratios for consistent processing
        self.sorted_ratios = sorted(self.ratios.keys())
    
    def _validate_points(self, points: List[FibonacciPoint]) -> bool:
        """
        Validate points for Fibonacci analysis.
        
        Args:
            points: List of points
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not points:
            logger.warning("No points provided for Fibonacci analysis")
            return False
        
        # Different Fibonacci types require different numbers of points
        min_points = 2
        if self.fibonacci_type == FibonacciType.EXTENSION:
            min_points = 3
        
        if len(points) < min_points:
            logger.warning(
                f"Insufficient points for {self.fibonacci_type.value} analysis. "
                f"Need at least {min_points}, got {len(points)}"
            )
            return False
        
        return True
    
    def _determine_direction(self, points: List[FibonacciPoint]) -> FibonacciDirection:
        """
        Determine the direction of the Fibonacci analysis.
        
        Args:
            points: List of points
            
        Returns:
            FibonacciDirection: Direction of analysis
        """
        # For most Fibonacci types, direction is determined by first two points
        start_price = points[0].price
        end_price = points[1].price
        
        if end_price > start_price:
            return FibonacciDirection.UPTREND
        else:
            return FibonacciDirection.DOWNTREND
    
    def _calculate_price_range(self, points: List[FibonacciPoint]) -> float:
        """
        Calculate the price range for Fibonacci analysis.
        
        Args:
            points: List of points
            
        Returns:
            float: Price range
        """
        # For most Fibonacci types, range is determined by first two points
        start_price = points[0].price
        end_price = points[1].price
        
        return abs(end_price - start_price)
    
    def _create_level(
        self, ratio: float, value: float, name: Optional[str] = None
    ) -> FibonacciLevel:
        """
        Create a Fibonacci level.
        
        Args:
            ratio: Fibonacci ratio
            value: Price value
            name: Level name (optional)
            
        Returns:
            FibonacciLevel: Fibonacci level
        """
        if name is None:
            name = self.ratios.get(ratio, f"{ratio:.3f}")
        
        is_key_level = ratio in self.key_levels
        
        description = None
        if is_key_level:
            if ratio == 0.0:
                description = "Starting point"
            elif ratio == 0.382:
                description = "Shallow retracement level"
            elif ratio == 0.5:
                description = "Mid-point retracement"
            elif ratio == 0.618:
                description = "Golden ratio retracement"
            elif ratio == 1.0:
                description = "Full retracement"
            elif ratio == 1.618:
                description = "Golden ratio extension"
            elif ratio == 2.618:
                description = "Strong extension target"
        
        return FibonacciLevel(
            ratio=ratio,
            value=value,
            name=name,
            is_key_level=is_key_level,
            description=description
        )
    
    def _format_levels(
        self, levels: List[FibonacciLevel], include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Format Fibonacci levels for output.
        
        Args:
            levels: List of Fibonacci levels
            include_metadata: Whether to include metadata
            
        Returns:
            List[Dict[str, Any]]: Formatted levels
        """
        result = []
        
        for level in levels:
            level_data = {
                "ratio": level.ratio,
                "value": level.value,
                "name": level.name
            }
            
            if include_metadata:
                level_data["is_key_level"] = level.is_key_level
                if level.description:
                    level_data["description"] = level.description
            
            result.append(level_data)
        
        return result