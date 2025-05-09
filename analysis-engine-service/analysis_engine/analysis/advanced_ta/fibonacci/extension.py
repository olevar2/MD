"""
Fibonacci Extension Module

This module implements Fibonacci extension analysis.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union

from core_foundations.utils.logger import get_logger
from core_foundations.exceptions.analysis_exceptions import FibonacciAnalysisError

from .base import (
    FibonacciBase,
    FibonacciType,
    FibonacciDirection,
    FibonacciPoint,
    FibonacciLevel
)

logger = get_logger(__name__)


class FibonacciExtension(FibonacciBase):
    """Fibonacci extension analysis."""
    
    def __init__(
        self,
        ratios: Optional[Dict[float, str]] = None,
        key_levels: Optional[List[float]] = None
    ):
        """
        Initialize Fibonacci extension.
        
        Args:
            ratios: Custom Fibonacci ratios (optional)
            key_levels: Custom key levels (optional)
        """
        # Default extension ratios focus on projections
        default_ratios = {
            0.0: "0%",
            0.618: "61.8%",
            1.0: "100%",
            1.272: "127.2%",
            1.414: "141.4%",
            1.618: "161.8%",
            2.0: "200%",
            2.618: "261.8%",
            3.618: "361.8%",
            4.236: "423.6%"
        }
        
        # Default key levels for extensions
        default_key_levels = [0.0, 1.0, 1.618, 2.618]
        
        super().__init__(
            fibonacci_type=FibonacciType.EXTENSION,
            ratios=ratios or default_ratios,
            key_levels=key_levels or default_key_levels
        )
    
    def calculate(
        self,
        points: List[FibonacciPoint],
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate Fibonacci extension levels.
        
        Args:
            points: List of points (requires at least 3 points)
            include_metadata: Whether to include metadata
            
        Returns:
            Dict[str, Any]: Extension levels and metadata
        """
        try:
            # Validate points
            if not self._validate_points(points):
                raise FibonacciAnalysisError("Invalid points for Fibonacci extension")
            
            # For extensions, we need at least 3 points
            if len(points) < 3:
                raise FibonacciAnalysisError("Fibonacci extension requires at least 3 points")
            
            # Points for extension:
            # Point 1: Initial swing start
            # Point 2: Initial swing end
            # Point 3: Retracement level
            point1 = points[0]
            point2 = points[1]
            point3 = points[2]
            
            # Determine direction of initial swing
            initial_direction = FibonacciDirection.UPTREND if point2.price > point1.price else FibonacciDirection.DOWNTREND
            
            # Calculate extension levels
            initial_range = abs(point2.price - point1.price)
            
            levels = []
            
            for ratio in self.sorted_ratios:
                if initial_direction == FibonacciDirection.UPTREND:
                    # For uptrend, extensions project up from retracement
                    value = point3.price + (ratio * initial_range)
                else:
                    # For downtrend, extensions project down from retracement
                    value = point3.price - (ratio * initial_range)
                
                level = self._create_level(ratio, value)
                levels.append(level)
            
            # Prepare result
            result = {
                "type": self.fibonacci_type.value,
                "direction": initial_direction.value,
                "point1": {
                    "index": point1.index,
                    "price": point1.price,
                    "timestamp": point1.timestamp
                },
                "point2": {
                    "index": point2.index,
                    "price": point2.price,
                    "timestamp": point2.timestamp
                },
                "point3": {
                    "index": point3.index,
                    "price": point3.price,
                    "timestamp": point3.timestamp
                },
                "levels": self._format_levels(levels, include_metadata)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci extension: {str(e)}", exc_info=True)
            raise FibonacciAnalysisError(f"Failed to calculate Fibonacci extension: {str(e)}")
    
    def find_nearest_level(
        self, price: float, levels: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Find the nearest Fibonacci level to a price.
        
        Args:
            price: Price to check
            levels: List of Fibonacci levels
            
        Returns:
            Dict[str, Any]: Nearest level with distance information
        """
        if not levels:
            return {
                "found": False,
                "message": "No levels available"
            }
        
        nearest_level = None
        min_distance = float('inf')
        
        for level in levels:
            distance = abs(level["value"] - price)
            
            if distance < min_distance:
                min_distance = distance
                nearest_level = level
        
        if nearest_level:
            # Calculate percentage distance
            level_price = nearest_level["value"]
            percent_distance = (abs(price - level_price) / level_price) * 100
            
            return {
                "found": True,
                "level": nearest_level,
                "distance": min_distance,
                "percent_distance": percent_distance
            }
        
        return {
            "found": False,
            "message": "Failed to find nearest level"
        }
    
    def is_at_level(
        self,
        price: float,
        levels: List[Dict[str, Any]],
        tolerance_percent: float = 0.5
    ) -> Dict[str, Any]:
        """
        Check if a price is at a Fibonacci level.
        
        Args:
            price: Price to check
            levels: List of Fibonacci levels
            tolerance_percent: Tolerance as percentage of level price
            
        Returns:
            Dict[str, Any]: Level information if at a level
        """
        nearest = self.find_nearest_level(price, levels)
        
        if not nearest["found"]:
            return {
                "at_level": False,
                "message": nearest["message"]
            }
        
        # Check if within tolerance
        if nearest["percent_distance"] <= tolerance_percent:
            return {
                "at_level": True,
                "level": nearest["level"],
                "distance": nearest["distance"],
                "percent_distance": nearest["percent_distance"]
            }
        
        return {
            "at_level": False,
            "nearest_level": nearest["level"],
            "percent_distance": nearest["percent_distance"]
        }