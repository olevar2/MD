"""
Fibonacci Fans Module

This module implements Fibonacci fans analysis.
"""
import logging
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from core_foundations.utils.logger import get_logger
from core_foundations.exceptions.analysis_exceptions import FibonacciAnalysisError
from .base import FibonacciBase, FibonacciType, FibonacciDirection, FibonacciPoint, FibonacciLevel
logger = get_logger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class FibonacciFans(FibonacciBase):
    """Fibonacci fans analysis."""

    def __init__(self, ratios: Optional[Dict[float, str]]=None, key_levels:
        Optional[List[float]]=None):
        """
        Initialize Fibonacci fans.
        
        Args:
            ratios: Custom Fibonacci ratios (optional)
            key_levels: Custom key levels (optional)
        """
        default_ratios = {(0.236): '23.6%', (0.382): '38.2%', (0.5): '50%',
            (0.618): '61.8%', (0.786): '78.6%'}
        default_key_levels = [0.382, 0.5, 0.618]
        super().__init__(fibonacci_type=FibonacciType.FANS, ratios=ratios or
            default_ratios, key_levels=key_levels or default_key_levels)

    @with_exception_handling
    def calculate(self, points: List[FibonacciPoint], include_metadata:
        bool=True, extend_right: int=100) ->Dict[str, Any]:
        """
        Calculate Fibonacci fans.
        
        Args:
            points: List of points (requires at least 2 points)
            include_metadata: Whether to include metadata
            extend_right: How far to extend fans to the right
            
        Returns:
            Dict[str, Any]: Fans and metadata
        """
        try:
            if not self._validate_points(points):
                raise FibonacciAnalysisError(
                    'Invalid points for Fibonacci fans')
            direction = self._determine_direction(points)
            price_range = self._calculate_price_range(points)
            start_point = points[0]
            end_point = points[1]
            time_range = end_point.index - start_point.index
            fans = []
            for ratio in self.sorted_ratios:
                if direction == FibonacciDirection.UPTREND:
                    fan_price = end_point.price - ratio * price_range
                else:
                    fan_price = end_point.price + ratio * price_range
                if time_range != 0:
                    slope = (fan_price - start_point.price) / time_range
                else:
                    slope = 0
                extended_index = end_point.index + extend_right
                extended_price = start_point.price + slope * (extended_index -
                    start_point.index)
                fan = {'ratio': ratio, 'name': self.ratios.get(ratio,
                    f'{ratio:.3f}'), 'start_point': {'index': start_point.
                    index, 'price': start_point.price, 'timestamp':
                    start_point.timestamp}, 'end_point': {'index':
                    end_point.index, 'price': fan_price}, 'extended_point':
                    {'index': extended_index, 'price': extended_price},
                    'slope': slope, 'is_key_level': ratio in self.key_levels}
                if fan['is_key_level'] and include_metadata:
                    if ratio == 0.382:
                        fan['description'] = 'Shallow fan line'
                    elif ratio == 0.5:
                        fan['description'] = 'Mid-point fan line'
                    elif ratio == 0.618:
                        fan['description'] = 'Golden ratio fan line'
                fans.append(fan)
            result = {'type': self.fibonacci_type.value, 'direction':
                direction.value, 'start_point': {'index': start_point.index,
                'price': start_point.price, 'timestamp': start_point.
                timestamp}, 'end_point': {'index': end_point.index, 'price':
                end_point.price, 'timestamp': end_point.timestamp}, 'fans':
                fans, 'extend_right': extend_right}
            return result
        except Exception as e:
            logger.error(f'Error calculating Fibonacci fans: {str(e)}',
                exc_info=True)
            raise FibonacciAnalysisError(
                f'Failed to calculate Fibonacci fans: {str(e)}')

    @with_resilience('get_price_at_index')
    def get_price_at_index(self, fan: Dict[str, Any], index: int) ->float:
        """
        Get price at a specific index on a fan line.
        
        Args:
            fan: Fan line data
            index: Index to get price at
            
        Returns:
            float: Price at index
        """
        start_point = fan['start_point']
        slope = fan['slope']
        price = start_point['price'] + slope * (index - start_point['index'])
        return price

    @with_exception_handling
    def is_price_at_fan(self, price: float, index: int, fans_data: Dict[str,
        Any], tolerance_percent: float=0.5) ->Dict[str, Any]:
        """
        Check if a price point is at a Fibonacci fan line.
        
        Args:
            price: Price to check
            index: Index to check
            fans_data: Fans data from calculate method
            tolerance_percent: Tolerance as percentage of price
            
        Returns:
            Dict[str, Any]: Fan information if at a fan line
        """
        try:
            nearest_fan = None
            min_price_diff = float('inf')
            for fan in fans_data['fans']:
                expected_price = self.get_price_at_index(fan, index)
                price_diff = abs(price - expected_price)
                if price_diff < min_price_diff:
                    min_price_diff = price_diff
                    nearest_fan = fan
            if nearest_fan:
                expected_price = self.get_price_at_index(nearest_fan, index)
                percent_diff = min_price_diff / expected_price * 100
                if percent_diff <= tolerance_percent:
                    return {'at_fan': True, 'fan': nearest_fan,
                        'price_diff': min_price_diff, 'percent_diff':
                        percent_diff, 'expected_price': expected_price}
                else:
                    return {'at_fan': False, 'nearest_fan': nearest_fan,
                        'price_diff': min_price_diff, 'percent_diff':
                        percent_diff, 'expected_price': expected_price}
            return {'at_fan': False, 'message': 'No fans available'}
        except Exception as e:
            logger.error(f'Error checking if price is at fan: {str(e)}',
                exc_info=True)
            return {'at_fan': False, 'error': str(e)}
