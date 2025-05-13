"""
Fibonacci Retracement Module

This module implements Fibonacci retracement analysis.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from core_foundations.utils.logger import get_logger
from core_foundations.exceptions.analysis_exceptions import FibonacciAnalysisError
from .base import FibonacciBase, FibonacciType, FibonacciDirection, FibonacciPoint, FibonacciLevel
logger = get_logger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class FibonacciRetracement(FibonacciBase):
    """Fibonacci retracement analysis."""

    def __init__(self, ratios: Optional[Dict[float, str]]=None, key_levels:
        Optional[List[float]]=None):
        """
        Initialize Fibonacci retracement.
        
        Args:
            ratios: Custom Fibonacci ratios (optional)
            key_levels: Custom key levels (optional)
        """
        super().__init__(fibonacci_type=FibonacciType.RETRACEMENT, ratios=
            ratios, key_levels=key_levels)

    @with_exception_handling
    def calculate(self, points: List[FibonacciPoint], include_metadata:
        bool=True) ->Dict[str, Any]:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            points: List of points (requires at least 2 points)
            include_metadata: Whether to include metadata
            
        Returns:
            Dict[str, Any]: Retracement levels and metadata
        """
        try:
            if not self._validate_points(points):
                raise FibonacciAnalysisError(
                    'Invalid points for Fibonacci retracement')
            direction = self._determine_direction(points)
            price_range = self._calculate_price_range(points)
            start_price = points[0].price
            end_price = points[1].price
            levels = []
            for ratio in self.sorted_ratios:
                if direction == FibonacciDirection.UPTREND:
                    value = end_price - ratio * price_range
                else:
                    value = end_price + ratio * price_range
                level = self._create_level(ratio, value)
                levels.append(level)
            result = {'type': self.fibonacci_type.value, 'direction':
                direction.value, 'start_point': {'index': points[0].index,
                'price': points[0].price, 'timestamp': points[0].timestamp},
                'end_point': {'index': points[1].index, 'price': points[1].
                price, 'timestamp': points[1].timestamp}, 'levels': self.
                _format_levels(levels, include_metadata)}
            return result
        except Exception as e:
            logger.error(f'Error calculating Fibonacci retracement: {str(e)}',
                exc_info=True)
            raise FibonacciAnalysisError(
                f'Failed to calculate Fibonacci retracement: {str(e)}')

    def find_nearest_level(self, price: float, levels: List[Dict[str, Any]]
        ) ->Dict[str, Any]:
        """
        Find the nearest Fibonacci level to a price.
        
        Args:
            price: Price to check
            levels: List of Fibonacci levels
            
        Returns:
            Dict[str, Any]: Nearest level with distance information
        """
        if not levels:
            return {'found': False, 'message': 'No levels available'}
        nearest_level = None
        min_distance = float('inf')
        for level in levels:
            distance = abs(level['value'] - price)
            if distance < min_distance:
                min_distance = distance
                nearest_level = level
        if nearest_level:
            level_price = nearest_level['value']
            percent_distance = abs(price - level_price) / level_price * 100
            return {'found': True, 'level': nearest_level, 'distance':
                min_distance, 'percent_distance': percent_distance}
        return {'found': False, 'message': 'Failed to find nearest level'}

    def is_at_level(self, price: float, levels: List[Dict[str, Any]],
        tolerance_percent: float=0.5) ->Dict[str, Any]:
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
        if not nearest['found']:
            return {'at_level': False, 'message': nearest['message']}
        if nearest['percent_distance'] <= tolerance_percent:
            return {'at_level': True, 'level': nearest['level'], 'distance':
                nearest['distance'], 'percent_distance': nearest[
                'percent_distance']}
        return {'at_level': False, 'nearest_level': nearest['level'],
            'percent_distance': nearest['percent_distance']}
