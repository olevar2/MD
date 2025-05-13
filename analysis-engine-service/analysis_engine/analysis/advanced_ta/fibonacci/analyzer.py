"""
Fibonacci Analyzer Module

This module implements a combined analyzer for all Fibonacci tools.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from core_foundations.utils.logger import get_logger
from core_foundations.exceptions.analysis_exceptions import FibonacciAnalysisError
from .base import FibonacciBase, FibonacciType, FibonacciDirection, FibonacciPoint, FibonacciLevel
from .retracement import FibonacciRetracement
from .extension import FibonacciExtension
from .arcs import FibonacciArcs
from .fans import FibonacciFans
from .time_zones import FibonacciTimeZones
logger = get_logger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class FibonacciAnalyzer:
    """Combined analyzer for all Fibonacci tools."""

    def __init__(self):
        """Initialize the Fibonacci analyzer."""
        self.retracement = FibonacciRetracement()
        self.extension = FibonacciExtension()
        self.arcs = FibonacciArcs()
        self.fans = FibonacciFans()
        self.time_zones = FibonacciTimeZones()
        logger.info('FibonacciAnalyzer initialized')

    @with_resilience('create_point')
    def create_point(self, index: int, price: float, timestamp: str
        ) ->FibonacciPoint:
        """
        Create a Fibonacci point.
        
        Args:
            index: Index of the point
            price: Price at the point
            timestamp: Timestamp of the point
            
        Returns:
            FibonacciPoint: Fibonacci point
        """
        return FibonacciPoint(index=index, price=price, timestamp=timestamp)

    @with_analysis_resilience('analyze_retracement')
    @with_exception_handling
    def analyze_retracement(self, points: List[Dict[str, Any]],
        include_metadata: bool=True) ->Dict[str, Any]:
        """
        Analyze Fibonacci retracement.
        
        Args:
            points: List of points (requires at least 2 points)
            include_metadata: Whether to include metadata
            
        Returns:
            Dict[str, Any]: Retracement analysis
        """
        try:
            fib_points = [self.create_point(index=p.get('index', i), price=
                p['price'], timestamp=p.get('timestamp', str(i))) for i, p in
                enumerate(points)]
            return self.retracement.calculate(fib_points, include_metadata)
        except Exception as e:
            logger.error(f'Error analyzing Fibonacci retracement: {str(e)}',
                exc_info=True)
            raise FibonacciAnalysisError(
                f'Failed to analyze Fibonacci retracement: {str(e)}')

    @with_analysis_resilience('analyze_extension')
    @with_exception_handling
    def analyze_extension(self, points: List[Dict[str, Any]],
        include_metadata: bool=True) ->Dict[str, Any]:
        """
        Analyze Fibonacci extension.
        
        Args:
            points: List of points (requires at least 3 points)
            include_metadata: Whether to include metadata
            
        Returns:
            Dict[str, Any]: Extension analysis
        """
        try:
            fib_points = [self.create_point(index=p.get('index', i), price=
                p['price'], timestamp=p.get('timestamp', str(i))) for i, p in
                enumerate(points)]
            return self.extension.calculate(fib_points, include_metadata)
        except Exception as e:
            logger.error(f'Error analyzing Fibonacci extension: {str(e)}',
                exc_info=True)
            raise FibonacciAnalysisError(
                f'Failed to analyze Fibonacci extension: {str(e)}')

    @with_analysis_resilience('analyze_arcs')
    @with_exception_handling
    def analyze_arcs(self, points: List[Dict[str, Any]], include_metadata:
        bool=True, price_scale: float=1.0, time_scale: float=1.0) ->Dict[
        str, Any]:
        """
        Analyze Fibonacci arcs.
        
        Args:
            points: List of points (requires at least 2 points)
            include_metadata: Whether to include metadata
            price_scale: Scale factor for price axis
            time_scale: Scale factor for time axis
            
        Returns:
            Dict[str, Any]: Arcs analysis
        """
        try:
            fib_points = [self.create_point(index=p.get('index', i), price=
                p['price'], timestamp=p.get('timestamp', str(i))) for i, p in
                enumerate(points)]
            return self.arcs.calculate(fib_points, include_metadata,
                price_scale, time_scale)
        except Exception as e:
            logger.error(f'Error analyzing Fibonacci arcs: {str(e)}',
                exc_info=True)
            raise FibonacciAnalysisError(
                f'Failed to analyze Fibonacci arcs: {str(e)}')

    @with_analysis_resilience('analyze_fans')
    @with_exception_handling
    def analyze_fans(self, points: List[Dict[str, Any]], include_metadata:
        bool=True, extend_right: int=100) ->Dict[str, Any]:
        """
        Analyze Fibonacci fans.
        
        Args:
            points: List of points (requires at least 2 points)
            include_metadata: Whether to include metadata
            extend_right: How far to extend fans to the right
            
        Returns:
            Dict[str, Any]: Fans analysis
        """
        try:
            fib_points = [self.create_point(index=p.get('index', i), price=
                p['price'], timestamp=p.get('timestamp', str(i))) for i, p in
                enumerate(points)]
            return self.fans.calculate(fib_points, include_metadata,
                extend_right)
        except Exception as e:
            logger.error(f'Error analyzing Fibonacci fans: {str(e)}',
                exc_info=True)
            raise FibonacciAnalysisError(
                f'Failed to analyze Fibonacci fans: {str(e)}')

    @with_analysis_resilience('analyze_time_zones')
    @with_exception_handling
    def analyze_time_zones(self, points: List[Dict[str, Any]],
        include_metadata: bool=True, time_unit: str='candle', max_zones: int=10
        ) ->Dict[str, Any]:
        """
        Analyze Fibonacci time zones.
        
        Args:
            points: List of points (requires at least 1 point)
            include_metadata: Whether to include metadata
            time_unit: Unit for time zones
            max_zones: Maximum number of zones to calculate
            
        Returns:
            Dict[str, Any]: Time zones analysis
        """
        try:
            fib_points = [self.create_point(index=p.get('index', i), price=
                p['price'], timestamp=p.get('timestamp', str(i))) for i, p in
                enumerate(points)]
            time_zones = FibonacciTimeZones(max_zones=max_zones)
            return time_zones.calculate(fib_points, include_metadata, time_unit
                )
        except Exception as e:
            logger.error(f'Error analyzing Fibonacci time zones: {str(e)}',
                exc_info=True)
            raise FibonacciAnalysisError(
                f'Failed to analyze Fibonacci time zones: {str(e)}')

    @with_analysis_resilience('analyze_all')
    @with_exception_handling
    def analyze_all(self, swing_points: List[Dict[str, Any]],
        include_metadata: bool=True) ->Dict[str, Any]:
        """
        Analyze all Fibonacci tools.
        
        Args:
            swing_points: List of swing points
            include_metadata: Whether to include metadata
            
        Returns:
            Dict[str, Any]: Combined analysis
        """
        try:
            result = {'swing_points': swing_points, 'analysis': {}}
            if len(swing_points) >= 2:
                result['analysis']['retracement'] = self.analyze_retracement(
                    swing_points[:2], include_metadata)
                result['analysis']['fans'] = self.analyze_fans(swing_points
                    [:2], include_metadata)
                result['analysis']['arcs'] = self.analyze_arcs(swing_points
                    [:2], include_metadata)
            if len(swing_points) >= 3:
                result['analysis']['extension'] = self.analyze_extension(
                    swing_points[:3], include_metadata)
            if swing_points:
                result['analysis']['time_zones'] = self.analyze_time_zones([
                    swing_points[0]], include_metadata)
            return result
        except Exception as e:
            logger.error(f'Error analyzing all Fibonacci tools: {str(e)}',
                exc_info=True)
            raise FibonacciAnalysisError(
                f'Failed to analyze all Fibonacci tools: {str(e)}')

    @with_resilience('check_price_at_levels')
    @with_exception_handling
    def check_price_at_levels(self, price: float, index: int,
        analysis_result: Dict[str, Any], tolerance_percent: float=0.5) ->Dict[
        str, Any]:
        """
        Check if a price is at any Fibonacci level.
        
        Args:
            price: Price to check
            index: Index to check
            analysis_result: Result from analyze_all method
            tolerance_percent: Tolerance as percentage
            
        Returns:
            Dict[str, Any]: Results for each tool
        """
        try:
            result = {'price': price, 'index': index, 'at_any_level': False,
                'tools': {}}
            if 'retracement' in analysis_result['analysis']:
                retracement_data = analysis_result['analysis']['retracement']
                retracement_check = self.retracement.is_at_level(price,
                    retracement_data['levels'], tolerance_percent)
                result['tools']['retracement'] = retracement_check
                if retracement_check.get('at_level', False):
                    result['at_any_level'] = True
            if 'extension' in analysis_result['analysis']:
                extension_data = analysis_result['analysis']['extension']
                extension_check = self.extension.is_at_level(price,
                    extension_data['levels'], tolerance_percent)
                result['tools']['extension'] = extension_check
                if extension_check.get('at_level', False):
                    result['at_any_level'] = True
            if 'fans' in analysis_result['analysis']:
                fans_data = analysis_result['analysis']['fans']
                fans_check = self.fans.is_price_at_fan(price, index,
                    fans_data, tolerance_percent)
                result['tools']['fans'] = fans_check
                if fans_check.get('at_fan', False):
                    result['at_any_level'] = True
            if 'arcs' in analysis_result['analysis']:
                arcs_data = analysis_result['analysis']['arcs']
                arcs_check = self.arcs.is_price_at_arc(price, index,
                    arcs_data, tolerance_percent)
                result['tools']['arcs'] = arcs_check
                if arcs_check.get('at_arc', False):
                    result['at_any_level'] = True
            if 'time_zones' in analysis_result['analysis']:
                time_zones_data = analysis_result['analysis']['time_zones']
                time_zones_check = self.time_zones.is_index_at_zone(index,
                    time_zones_data, tolerance=1)
                result['tools']['time_zones'] = time_zones_check
                if time_zones_check.get('at_zone', False):
                    result['at_any_level'] = True
            return result
        except Exception as e:
            logger.error(f'Error checking price at levels: {str(e)}',
                exc_info=True)
            return {'price': price, 'index': index, 'at_any_level': False,
                'error': str(e)}
