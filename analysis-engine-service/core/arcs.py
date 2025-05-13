"""
Fibonacci Arcs Module

This module implements Fibonacci arcs analysis.
"""
import logging
import math
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

class FibonacciArcs(FibonacciBase):
    """Fibonacci arcs analysis."""

    def __init__(self, ratios: Optional[Dict[float, str]]=None, key_levels:
        Optional[List[float]]=None):
        """
        Initialize Fibonacci arcs.
        
        Args:
            ratios: Custom Fibonacci ratios (optional)
            key_levels: Custom key levels (optional)
        """
        default_ratios = {(0.236): '23.6%', (0.382): '38.2%', (0.5): '50%',
            (0.618): '61.8%', (0.786): '78.6%'}
        default_key_levels = [0.382, 0.5, 0.618]
        super().__init__(fibonacci_type=FibonacciType.ARCS, ratios=ratios or
            default_ratios, key_levels=key_levels or default_key_levels)

    @with_exception_handling
    def calculate(self, points: List[FibonacciPoint], include_metadata:
        bool=True, price_scale: float=1.0, time_scale: float=1.0) ->Dict[
        str, Any]:
        """
        Calculate Fibonacci arcs.
        
        Args:
            points: List of points (requires at least 2 points)
            include_metadata: Whether to include metadata
            price_scale: Scale factor for price axis
            time_scale: Scale factor for time axis
            
        Returns:
            Dict[str, Any]: Arcs and metadata
        """
        try:
            if not self._validate_points(points):
                raise FibonacciAnalysisError(
                    'Invalid points for Fibonacci arcs')
            direction = self._determine_direction(points)
            price_range = self._calculate_price_range(points)
            center_point = points[0]
            end_point = points[1]
            radius = price_range
            arcs = []
            for ratio in self.sorted_ratios:
                arc_radius = ratio * radius
                arc = {'ratio': ratio, 'name': self.ratios.get(ratio,
                    f'{ratio:.3f}'), 'center': {'index': center_point.index,
                    'price': center_point.price, 'timestamp': center_point.
                    timestamp}, 'radius': arc_radius, 'is_key_level': ratio in
                    self.key_levels}
                if arc['is_key_level'] and include_metadata:
                    if ratio == 0.382:
                        arc['description'] = 'Shallow arc level'
                    elif ratio == 0.5:
                        arc['description'] = 'Mid-point arc'
                    elif ratio == 0.618:
                        arc['description'] = 'Golden ratio arc'
                arcs.append(arc)
            result = {'type': self.fibonacci_type.value, 'direction':
                direction.value, 'center_point': {'index': center_point.
                index, 'price': center_point.price, 'timestamp':
                center_point.timestamp}, 'end_point': {'index': end_point.
                index, 'price': end_point.price, 'timestamp': end_point.
                timestamp}, 'radius': radius, 'arcs': arcs, 'price_scale':
                price_scale, 'time_scale': time_scale}
            return result
        except Exception as e:
            logger.error(f'Error calculating Fibonacci arcs: {str(e)}',
                exc_info=True)
            raise FibonacciAnalysisError(
                f'Failed to calculate Fibonacci arcs: {str(e)}')

    @with_exception_handling
    def is_price_at_arc(self, price: float, time_index: int, arcs_data:
        Dict[str, Any], tolerance_percent: float=1.0) ->Dict[str, Any]:
        """
        Check if a price point is at a Fibonacci arc.
        
        Args:
            price: Price to check
            time_index: Time index to check
            arcs_data: Arcs data from calculate method
            tolerance_percent: Tolerance as percentage
            
        Returns:
            Dict[str, Any]: Arc information if at an arc
        """
        try:
            center_point = arcs_data['center_point']
            price_scale = arcs_data.get('price_scale', 1.0)
            time_scale = arcs_data.get('time_scale', 1.0)
            price_distance = abs(price - center_point['price']) * price_scale
            time_distance = abs(time_index - center_point['index']
                ) * time_scale
            actual_distance = math.sqrt(price_distance ** 2 + time_distance **
                2)
            nearest_arc = None
            min_distance_diff = float('inf')
            for arc in arcs_data['arcs']:
                arc_radius = arc['radius']
                distance_diff = abs(actual_distance - arc_radius)
                if distance_diff < min_distance_diff:
                    min_distance_diff = distance_diff
                    nearest_arc = arc
            if nearest_arc:
                percent_diff = min_distance_diff / nearest_arc['radius'] * 100
                if percent_diff <= tolerance_percent:
                    return {'at_arc': True, 'arc': nearest_arc,
                        'distance_diff': min_distance_diff, 'percent_diff':
                        percent_diff}
                else:
                    return {'at_arc': False, 'nearest_arc': nearest_arc,
                        'distance_diff': min_distance_diff, 'percent_diff':
                        percent_diff}
            return {'at_arc': False, 'message': 'No arcs available'}
        except Exception as e:
            logger.error(f'Error checking if price is at arc: {str(e)}',
                exc_info=True)
            return {'at_arc': False, 'error': str(e)}
