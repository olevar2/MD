"""
Fibonacci Time Zones Module

This module implements Fibonacci time zones analysis.
"""
import logging
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

class FibonacciTimeZones(FibonacciBase):
    """Fibonacci time zones analysis."""
    FIBONACCI_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 
        377, 610, 987]

    def __init__(self, max_zones: int=10, key_zones: Optional[List[int]]=None):
        """
        Initialize Fibonacci time zones.
        
        Args:
            max_zones: Maximum number of time zones to calculate
            key_zones: Custom key zones (optional)
        """
        ratios = {}
        for i, value in enumerate(self.FIBONACCI_SEQUENCE[:max_zones]):
            ratios[value] = f'Fib {i + 1}'
        default_key_zones = [1, 2, 3, 5, 8, 13, 21, 34]
        super().__init__(fibonacci_type=FibonacciType.TIME_ZONES, ratios=
            ratios, key_levels=key_zones or default_key_zones)
        self.max_zones = max_zones

    @with_exception_handling
    def calculate(self, points: List[FibonacciPoint], include_metadata:
        bool=True, time_unit: str='candle') ->Dict[str, Any]:
        """
        Calculate Fibonacci time zones.
        
        Args:
            points: List of points (requires at least 1 point)
            include_metadata: Whether to include metadata
            time_unit: Unit for time zones (candle, day, hour, etc.)
            
        Returns:
            Dict[str, Any]: Time zones and metadata
        """
        try:
            if not points or len(points) < 1:
                raise FibonacciAnalysisError(
                    'Invalid points for Fibonacci time zones')
            start_point = points[0]
            zones = []
            for i, value in enumerate(self.FIBONACCI_SEQUENCE[:self.max_zones]
                ):
                zone_index = start_point.index + value
                zone = {'sequence_number': i + 1, 'fibonacci_value': value,
                    'name': f'Fib {i + 1}', 'index': zone_index, 'offset':
                    value, 'is_key_zone': value in self.key_levels}
                if zone['is_key_zone'] and include_metadata:
                    if value == 1:
                        zone['description'] = 'First time zone'
                    elif value == 2:
                        zone['description'] = 'Second time zone'
                    elif value == 3:
                        zone['description'] = 'Third time zone'
                    elif value == 5:
                        zone['description'
                            ] = 'Fifth time zone (often significant)'
                    elif value == 8:
                        zone['description'
                            ] = 'Eighth time zone (often significant)'
                    elif value == 13:
                        zone['description'
                            ] = 'Thirteenth time zone (often significant)'
                zones.append(zone)
            result = {'type': self.fibonacci_type.value, 'start_point': {
                'index': start_point.index, 'price': start_point.price,
                'timestamp': start_point.timestamp}, 'zones': zones,
                'time_unit': time_unit}
            return result
        except Exception as e:
            logger.error(f'Error calculating Fibonacci time zones: {str(e)}',
                exc_info=True)
            raise FibonacciAnalysisError(
                f'Failed to calculate Fibonacci time zones: {str(e)}')

    @with_exception_handling
    def is_index_at_zone(self, index: int, zones_data: Dict[str, Any],
        tolerance: int=0) ->Dict[str, Any]:
        """
        Check if an index is at a Fibonacci time zone.
        
        Args:
            index: Index to check
            zones_data: Zones data from calculate method
            tolerance: Tolerance in index units
            
        Returns:
            Dict[str, Any]: Zone information if at a zone
        """
        try:
            for zone in zones_data['zones']:
                zone_index = zone['index']
                if abs(index - zone_index) <= tolerance:
                    return {'at_zone': True, 'zone': zone, 'distance': abs(
                        index - zone_index)}
            nearest_zone = None
            min_distance = float('inf')
            for zone in zones_data['zones']:
                distance = abs(index - zone['index'])
                if distance < min_distance:
                    min_distance = distance
                    nearest_zone = zone
            if nearest_zone:
                return {'at_zone': False, 'nearest_zone': nearest_zone,
                    'distance': min_distance}
            return {'at_zone': False, 'message': 'No zones available'}
        except Exception as e:
            logger.error(f'Error checking if index is at zone: {str(e)}',
                exc_info=True)
            return {'at_zone': False, 'error': str(e)}

    @with_resilience('get_next_zones')
    @with_exception_handling
    def get_next_zones(self, current_index: int, zones_data: Dict[str, Any],
        count: int=3) ->List[Dict[str, Any]]:
        """
        Get the next Fibonacci time zones from current index.
        
        Args:
            current_index: Current index
            zones_data: Zones data from calculate method
            count: Number of zones to return
            
        Returns:
            List[Dict[str, Any]]: Next time zones
        """
        try:
            future_zones = [zone for zone in zones_data['zones'] if zone[
                'index'] > current_index]
            future_zones.sort(key=lambda z: z['index'])
            return future_zones[:count]
        except Exception as e:
            logger.error(f'Error getting next zones: {str(e)}', exc_info=True)
            return []
