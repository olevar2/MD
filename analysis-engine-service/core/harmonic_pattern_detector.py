"""
Harmonic Pattern Detector

This module provides functionality to detect and validate various harmonic price patterns
including Gartley, Butterfly, Bat, Crab and other harmonic formations.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from enum import Enum
import math


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class HarmonicPatternType(str, Enum):
    """Types of harmonic patterns that can be detected"""
    GARTLEY = 'gartley'
    BUTTERFLY = 'butterfly'
    BAT = 'bat'
    CRAB = 'crab'
    SHARK = 'shark'
    CYPHER = 'cypher'
    THREE_DRIVE = 'three_drive'
    ABCD = 'abcd'


class PatternDirection(str, Enum):
    """Pattern direction"""
    BULLISH = 'bullish'
    BEARISH = 'bearish'


class PatternPoint:
    """Represents a pivot point in a harmonic pattern"""

    def __init__(self, index: int, price: float, timestamp: datetime):
        """
        Initialize a pattern point.
        
        Args:
            index: Index/position in the data
            price: Price value
            timestamp: Timestamp of the point
        """
        self.index = index
        self.price = price
        self.timestamp = timestamp


class HarmonicPattern:
    """Represents a detected harmonic pattern"""

    def __init__(self, pattern_type: HarmonicPatternType, direction:
        PatternDirection, points: Dict[str, PatternPoint], ratios: Dict[str,
        float], completion: float=1.0):
        """
        Initialize a harmonic pattern.
        
        Args:
            pattern_type: Type of harmonic pattern
            direction: Pattern direction (bullish/bearish)
            points: Dictionary mapping point names (X,A,B,C,D) to PatternPoint objects
            ratios: Dictionary of the pattern's Fibonacci ratios
            completion: Completion percentage (0.0-1.0)
        """
        self.pattern_type = pattern_type
        self.direction = direction
        self.points = points
        self.ratios = ratios
        self.completion = completion
        self.confidence = 0.0
        self.detected_at = datetime.utcnow()
        self._calculate_confidence()

    def _calculate_confidence(self) ->None:
        """Calculate pattern confidence based on ratio precision"""
        if not self.ratios:
            self.confidence = 0.0
            return
        ideal_ratios = PATTERN_IDEAL_RATIOS.get(self.pattern_type, {})
        if not ideal_ratios:
            self.confidence = 0.5
            return
        precision_scores = []
        for ratio_name, actual_ratio in self.ratios.items():
            if ratio_name not in ideal_ratios:
                continue
            ideal_ratio = ideal_ratios[ratio_name]
            tolerance = ideal_ratios.get(f'{ratio_name}_tolerance', 0.05)
            error = abs(actual_ratio - ideal_ratio)
            precision = max(0.0, 1.0 - error / tolerance)
            precision_scores.append(precision)
        if precision_scores:
            self.confidence = sum(precision_scores) / len(precision_scores)
        else:
            self.confidence = 0.5
        self.confidence *= self.completion

    @with_resilience('get_target_price')
    def get_target_price(self) ->Optional[float]:
        """Get the target price projection for this pattern"""
        if 'D' not in self.points:
            return None
        return self.points['D'].price

    @with_resilience('get_entry_price')
    def get_entry_price(self) ->Optional[float]:
        """Get the suggested entry price for this pattern"""
        if 'D' not in self.points:
            return None
        return self.points['D'].price

    @with_resilience('get_stop_loss_price')
    def get_stop_loss_price(self) ->Optional[float]:
        """Get the suggested stop loss price for this pattern"""
        if 'D' not in self.points or 'X' not in self.points:
            return None
        if self.direction == PatternDirection.BULLISH:
            return self.points['X'].price * 0.995
        else:
            return self.points['X'].price * 1.005

    def to_dict(self) ->Dict[str, Any]:
        """Convert pattern to dictionary representation"""
        return {'pattern_type': self.pattern_type, 'direction': self.
            direction, 'points': {name: {'price': point.price, 'timestamp':
            point.timestamp.isoformat()} for name, point in self.points.
            items()}, 'ratios': self.ratios, 'completion': self.completion,
            'confidence': self.confidence, 'target_price': self.
            get_target_price(), 'entry_price': self.get_entry_price(),
            'stop_loss': self.get_stop_loss_price(), 'detected_at': self.
            detected_at.isoformat()}


PATTERN_IDEAL_RATIOS = {HarmonicPatternType.GARTLEY: {'AB': 0.618,
    'AB_tolerance': 0.06, 'BC': 0.382, 'BC_tolerance': 0.04, 'CD': 1.272,
    'CD_tolerance': 0.1, 'XA': 1.0, 'XA_tolerance': 0.1},
    HarmonicPatternType.BUTTERFLY: {'AB': 0.786, 'AB_tolerance': 0.06, 'BC':
    0.382, 'BC_tolerance': 0.04, 'CD': 1.618, 'CD_tolerance': 0.12, 'XA': 
    1.0, 'XA_tolerance': 0.1}, HarmonicPatternType.BAT: {'AB': 0.5,
    'AB_tolerance': 0.05, 'BC': 0.382, 'BC_tolerance': 0.04, 'CD': 1.618,
    'CD_tolerance': 0.12, 'XA': 1.0, 'XA_tolerance': 0.1},
    HarmonicPatternType.CRAB: {'AB': 0.382, 'AB_tolerance': 0.04, 'BC': 
    0.382, 'BC_tolerance': 0.04, 'CD': 2.618, 'CD_tolerance': 0.2, 'XA': 
    1.0, 'XA_tolerance': 0.1}, HarmonicPatternType.SHARK: {'AB': 0.5,
    'AB_tolerance': 0.05, 'BC': 0.382, 'BC_tolerance': 0.04, 'CD': 1.618,
    'CD_tolerance': 0.12, 'XA': 1.13, 'XA_tolerance': 0.1},
    HarmonicPatternType.CYPHER: {'AB': 0.382, 'AB_tolerance': 0.04, 'BC': 
    0.618, 'BC_tolerance': 0.06, 'CD': 1.272, 'CD_tolerance': 0.1, 'XA': 
    1.0, 'XA_tolerance': 0.1}, HarmonicPatternType.ABCD: {'AB': 0.382,
    'AB_tolerance': 0.04, 'BC': 0.382, 'BC_tolerance': 0.04, 'CD': 1.272,
    'CD_tolerance': 0.1}}


class HarmonicPatternAnalyzer:
    """
    Detects and validates harmonic patterns in price data.
    
    This detector identifies common harmonic patterns based on Fibonacci
    relationships between price points, validating them against expected ratios.
    """

    def __init__(self, min_pattern_bars: int=10, max_pattern_bars: int=100,
        completion_threshold: float=0.9):
        """
        Initialize the harmonic pattern detector.
        
        Args:
            min_pattern_bars: Minimum bar span for valid patterns
            max_pattern_bars: Maximum bar span for valid patterns
            completion_threshold: Threshold for pattern completion (0.0-1.0)
        """
        self.logger = logging.getLogger(__name__)
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        self.completion_threshold = completion_threshold

    def detect_patterns(self, price_data: pd.DataFrame, pattern_types:
        Optional[List[HarmonicPatternType]]=None) ->List[Dict[str, Any]]:
        """
        Detect harmonic patterns in the provided price data.
        
        Args:
            price_data: OHLCV price data as pandas DataFrame
            pattern_types: Optional list of specific pattern types to detect
                          (if None, detects all pattern types)
                          
        Returns:
            List of detected patterns as dictionaries
        """
        if price_data.empty:
            return []
        if pattern_types is None:
            pattern_types = list(HarmonicPatternType)
        pivots = self._identify_pivot_points(price_data)
        patterns = []
        for pattern_type in pattern_types:
            if pattern_type == HarmonicPatternType.GARTLEY:
                patterns.extend(self._detect_gartley_patterns(pivots,
                    price_data))
            elif pattern_type == HarmonicPatternType.BUTTERFLY:
                patterns.extend(self._detect_butterfly_patterns(pivots,
                    price_data))
            elif pattern_type == HarmonicPatternType.BAT:
                patterns.extend(self._detect_bat_patterns(pivots, price_data))
            elif pattern_type == HarmonicPatternType.CRAB:
                patterns.extend(self._detect_crab_patterns(pivots, price_data))
            elif pattern_type == HarmonicPatternType.SHARK:
                patterns.extend(self._detect_shark_patterns(pivots, price_data)
                    )
            elif pattern_type == HarmonicPatternType.CYPHER:
                patterns.extend(self._detect_cypher_patterns(pivots,
                    price_data))
            elif pattern_type == HarmonicPatternType.THREE_DRIVE:
                patterns.extend(self._detect_three_drive_patterns(pivots,
                    price_data))
            elif pattern_type == HarmonicPatternType.ABCD:
                patterns.extend(self._detect_abcd_patterns(pivots, price_data))
        return [pattern.to_dict() for pattern in patterns]

    def _identify_pivot_points(self, price_data: pd.DataFrame, window: int=5
        ) ->List[PatternPoint]:
        """
        Identify pivot points (highs and lows) in price data.
        
        Args:
            price_data: OHLCV price data
            window: Window size for pivot detection
            
        Returns:
            List of pivot points
        """
        pivots = []
        if len(price_data) < 2 * window + 1:
            return pivots
        highs = price_data['high'].values
        lows = price_data['low'].values
        for i in range(window, len(price_data) - window):
            is_pivot_high = True
            for j in range(i - window, i):
                if highs[j] > highs[i]:
                    is_pivot_high = False
                    break
            for j in range(i + 1, i + window + 1):
                if highs[j] > highs[i]:
                    is_pivot_high = False
                    break
            if is_pivot_high:
                pivots.append(PatternPoint(index=i, price=highs[i],
                    timestamp=price_data.index[i]))
        for i in range(window, len(price_data) - window):
            is_pivot_low = True
            for j in range(i - window, i):
                if lows[j] < lows[i]:
                    is_pivot_low = False
                    break
            for j in range(i + 1, i + window + 1):
                if lows[j] < lows[i]:
                    is_pivot_low = False
                    break
            if is_pivot_low:
                pivots.append(PatternPoint(index=i, price=lows[i],
                    timestamp=price_data.index[i]))
        pivots.sort(key=lambda x: x.index)
        return pivots

    def _calculate_pattern_ratios(self, points: Dict[str, PatternPoint]
        ) ->Dict[str, float]:
        """
        Calculate Fibonacci ratios for pattern points.
        
        Args:
            points: Dictionary mapping point names to PatternPoint objects
            
        Returns:
            Dictionary of calculated ratios
        """
        ratios = {}
        if 'X' in points and 'A' in points:
            xa_range = abs(points['A'].price - points['X'].price)
            ratios['XA'] = xa_range
        if 'A' in points and 'B' in points and 'XA' in ratios and ratios['XA'
            ] != 0:
            ab_range = abs(points['B'].price - points['A'].price)
            ratios['AB'] = ab_range / ratios['XA']
        if 'B' in points and 'C' in points and 'AB' in ratios and ratios['AB'
            ] != 0:
            bc_range = abs(points['C'].price - points['B'].price)
            ab_range = abs(points['B'].price - points['A'].price)
            ratios['BC'] = bc_range / ab_range
        if 'C' in points and 'D' in points and 'BC' in ratios and ratios['BC'
            ] != 0:
            cd_range = abs(points['D'].price - points['C'].price)
            bc_range = abs(points['C'].price - points['B'].price)
            ratios['CD'] = cd_range / bc_range
        return ratios

    def _detect_gartley_patterns(self, pivots: List[PatternPoint],
        price_data: pd.DataFrame) ->List[HarmonicPattern]:
        """
        Detect Gartley patterns among the given pivot points.
        
        Args:
            pivots: List of pivot points
            price_data: OHLCV price data
            
        Returns:
            List of detected Gartley patterns
        """
        patterns = []
        if len(pivots) < 4:
            return patterns
        for i in range(len(pivots) - 3):
            point_x = pivots[i]
            point_a = pivots[i + 1]
            point_b = pivots[i + 2]
            point_c = pivots[i + 3]
            point_d = self._calculate_potential_d_point(HarmonicPatternType
                .GARTLEY, point_x, point_a, point_b, point_c)
            span = point_c.index - point_x.index
            if span < self.min_pattern_bars or span > self.max_pattern_bars:
                continue
            is_bullish = point_a.price < point_x.price
            latest_price = price_data.iloc[-1]['close']
            points = {'X': point_x, 'A': point_a, 'B': point_b, 'C': point_c}
            if point_d is not None:
                points['D'] = point_d
            ratios = self._calculate_pattern_ratios(points)
            if not self._validate_pattern_ratios(ratios,
                HarmonicPatternType.GARTLEY):
                continue
            completion = 1.0
            if point_d is None:
                if is_bullish:
                    d_projection = self._project_d_point(HarmonicPatternType
                        .GARTLEY, True, point_x, point_a, point_b, point_c)
                    if d_projection > latest_price:
                        completion = (latest_price - point_c.price) / (
                            d_projection - point_c.price)
                else:
                    d_projection = self._project_d_point(HarmonicPatternType
                        .GARTLEY, False, point_x, point_a, point_b, point_c)
                    if d_projection < latest_price:
                        completion = (point_c.price - latest_price) / (point_c
                            .price - d_projection)
                completion = max(0.0, min(1.0, completion))
            if completion < self.completion_threshold:
                continue
            pattern = HarmonicPattern(pattern_type=HarmonicPatternType.
                GARTLEY, direction=PatternDirection.BULLISH if is_bullish else
                PatternDirection.BEARISH, points=points, ratios=ratios,
                completion=completion)
            patterns.append(pattern)
        return patterns

    def _detect_butterfly_patterns(self, pivots: List[PatternPoint],
        price_data: pd.DataFrame) ->List[HarmonicPattern]:
        """
        Detect Butterfly patterns among the given pivot points.
        
        Args:
            pivots: List of pivot points
            price_data: OHLCV price data
            
        Returns:
            List of detected Butterfly patterns
        """
        patterns = []
        if len(pivots) < 4:
            return patterns
        for i in range(len(pivots) - 3):
            point_x = pivots[i]
            point_a = pivots[i + 1]
            point_b = pivots[i + 2]
            point_c = pivots[i + 3]
            point_d = self._calculate_potential_d_point(HarmonicPatternType
                .BUTTERFLY, point_x, point_a, point_b, point_c)
            span = point_c.index - point_x.index
            if span < self.min_pattern_bars or span > self.max_pattern_bars:
                continue
            is_bullish = point_a.price < point_x.price
            latest_price = price_data.iloc[-1]['close']
            points = {'X': point_x, 'A': point_a, 'B': point_b, 'C': point_c}
            if point_d is not None:
                points['D'] = point_d
            ratios = self._calculate_pattern_ratios(points)
            if not self._validate_pattern_ratios(ratios,
                HarmonicPatternType.BUTTERFLY):
                continue
            completion = 1.0
            if point_d is None:
                if is_bullish:
                    d_projection = self._project_d_point(HarmonicPatternType
                        .BUTTERFLY, True, point_x, point_a, point_b, point_c)
                    if d_projection > latest_price:
                        completion = (latest_price - point_c.price) / (
                            d_projection - point_c.price)
                else:
                    d_projection = self._project_d_point(HarmonicPatternType
                        .BUTTERFLY, False, point_x, point_a, point_b, point_c)
                    if d_projection < latest_price:
                        completion = (point_c.price - latest_price) / (point_c
                            .price - d_projection)
                completion = max(0.0, min(1.0, completion))
            if completion < self.completion_threshold:
                continue
            pattern = HarmonicPattern(pattern_type=HarmonicPatternType.
                BUTTERFLY, direction=PatternDirection.BULLISH if is_bullish
                 else PatternDirection.BEARISH, points=points, ratios=
                ratios, completion=completion)
            patterns.append(pattern)
        return patterns

    def _detect_bat_patterns(self, pivots: List[PatternPoint], price_data:
        pd.DataFrame) ->List[HarmonicPattern]:
        """
        Detect Bat patterns among the given pivot points.
        
        Args:
            pivots: List of pivot points
            price_data: OHLCV price data
            
        Returns:
            List of detected Bat patterns
        """
        patterns = []
        if len(pivots) < 4:
            return patterns
        for i in range(len(pivots) - 3):
            point_x = pivots[i]
            point_a = pivots[i + 1]
            point_b = pivots[i + 2]
            point_c = pivots[i + 3]
            point_d = self._calculate_potential_d_point(HarmonicPatternType
                .BAT, point_x, point_a, point_b, point_c)
            span = point_c.index - point_x.index
            if span < self.min_pattern_bars or span > self.max_pattern_bars:
                continue
            is_bullish = point_a.price < point_x.price
            latest_price = price_data.iloc[-1]['close']
            points = {'X': point_x, 'A': point_a, 'B': point_b, 'C': point_c}
            if point_d is not None:
                points['D'] = point_d
            ratios = self._calculate_pattern_ratios(points)
            if not self._validate_pattern_ratios(ratios,
                HarmonicPatternType.BAT):
                continue
            completion = 1.0
            if point_d is None:
                if is_bullish:
                    d_projection = self._project_d_point(HarmonicPatternType
                        .BAT, True, point_x, point_a, point_b, point_c)
                    if d_projection > latest_price:
                        completion = (latest_price - point_c.price) / (
                            d_projection - point_c.price)
                else:
                    d_projection = self._project_d_point(HarmonicPatternType
                        .BAT, False, point_x, point_a, point_b, point_c)
                    if d_projection < latest_price:
                        completion = (point_c.price - latest_price) / (point_c
                            .price - d_projection)
                completion = max(0.0, min(1.0, completion))
            if completion < self.completion_threshold:
                continue
            pattern = HarmonicPattern(pattern_type=HarmonicPatternType.BAT,
                direction=PatternDirection.BULLISH if is_bullish else
                PatternDirection.BEARISH, points=points, ratios=ratios,
                completion=completion)
            patterns.append(pattern)
        return patterns

    def _detect_crab_patterns(self, pivots: List[PatternPoint], price_data:
        pd.DataFrame) ->List[HarmonicPattern]:
        """
        Detect Crab patterns among the given pivot points.
        
        Args:
            pivots: List of pivot points
            price_data: OHLCV price data
            
        Returns:
            List of detected Crab patterns
        """
        patterns = []
        if len(pivots) < 4:
            return patterns
        for i in range(len(pivots) - 3):
            point_x = pivots[i]
            point_a = pivots[i + 1]
            point_b = pivots[i + 2]
            point_c = pivots[i + 3]
            point_d = self._calculate_potential_d_point(HarmonicPatternType
                .CRAB, point_x, point_a, point_b, point_c)
            span = point_c.index - point_x.index
            if span < self.min_pattern_bars or span > self.max_pattern_bars:
                continue
            is_bullish = point_a.price < point_x.price
            latest_price = price_data.iloc[-1]['close']
            points = {'X': point_x, 'A': point_a, 'B': point_b, 'C': point_c}
            if point_d is not None:
                points['D'] = point_d
            ratios = self._calculate_pattern_ratios(points)
            if not self._validate_pattern_ratios(ratios,
                HarmonicPatternType.CRAB):
                continue
            completion = 1.0
            if point_d is None:
                if is_bullish:
                    d_projection = self._project_d_point(HarmonicPatternType
                        .CRAB, True, point_x, point_a, point_b, point_c)
                    if d_projection > latest_price:
                        completion = (latest_price - point_c.price) / (
                            d_projection - point_c.price)
                else:
                    d_projection = self._project_d_point(HarmonicPatternType
                        .CRAB, False, point_x, point_a, point_b, point_c)
                    if d_projection < latest_price:
                        completion = (point_c.price - latest_price) / (point_c
                            .price - d_projection)
                completion = max(0.0, min(1.0, completion))
            if completion < self.completion_threshold:
                continue
            pattern = HarmonicPattern(pattern_type=HarmonicPatternType.CRAB,
                direction=PatternDirection.BULLISH if is_bullish else
                PatternDirection.BEARISH, points=points, ratios=ratios,
                completion=completion)
            patterns.append(pattern)
        return patterns

    def _detect_shark_patterns(self, pivots: List[PatternPoint], price_data:
        pd.DataFrame) ->List[HarmonicPattern]:
        """
        Detect Shark patterns among the given pivot points.
        
        Args:
            pivots: List of pivot points
            price_data: OHLCV price data
            
        Returns:
            List of detected Shark patterns
        """
        return []

    def _detect_cypher_patterns(self, pivots: List[PatternPoint],
        price_data: pd.DataFrame) ->List[HarmonicPattern]:
        """
        Detect Cypher patterns among the given pivot points.
        
        Args:
            pivots: List of pivot points
            price_data: OHLCV price data
            
        Returns:
            List of detected Cypher patterns
        """
        return []

    def _detect_three_drive_patterns(self, pivots: List[PatternPoint],
        price_data: pd.DataFrame) ->List[HarmonicPattern]:
        """
        Detect Three Drive patterns among the given pivot points.
        
        Args:
            pivots: List of pivot points
            price_data: OHLCV price data
            
        Returns:
            List of detected Three Drive patterns
        """
        return []

    def _detect_abcd_patterns(self, pivots: List[PatternPoint], price_data:
        pd.DataFrame) ->List[HarmonicPattern]:
        """
        Detect ABCD patterns among the given pivot points.
        
        Args:
            pivots: List of pivot points
            price_data: OHLCV price data
            
        Returns:
            List of detected ABCD patterns
        """
        patterns = []
        if len(pivots) < 3:
            return patterns
        for i in range(len(pivots) - 2):
            point_a = pivots[i]
            point_b = pivots[i + 1]
            point_c = pivots[i + 2]
            point_d = self._calculate_potential_abcd_point(point_a, point_b,
                point_c)
            span = point_c.index - point_a.index
            if span < self.min_pattern_bars or span > self.max_pattern_bars:
                continue
            is_bullish = point_b.price < point_a.price
            latest_price = price_data.iloc[-1]['close']
            points = {'A': point_a, 'B': point_b, 'C': point_c}
            if point_d is not None:
                points['D'] = point_d
            ratios = {}
            ab_range = abs(point_b.price - point_a.price)
            ratios['AB'] = ab_range
            bc_range = abs(point_c.price - point_b.price)
            ratios['BC'] = bc_range / ab_range if ab_range != 0 else 0
            if point_d is not None:
                cd_range = abs(point_d.price - point_c.price)
                ratios['CD'] = cd_range / bc_range if bc_range != 0 else 0
            if not self._validate_pattern_ratios(ratios,
                HarmonicPatternType.ABCD):
                continue
            completion = 1.0
            if point_d is None:
                if is_bullish:
                    d_projection = self._project_abcd_point(True, point_a,
                        point_b, point_c)
                    if d_projection > latest_price:
                        completion = (latest_price - point_c.price) / (
                            d_projection - point_c.price)
                else:
                    d_projection = self._project_abcd_point(False, point_a,
                        point_b, point_c)
                    if d_projection < latest_price:
                        completion = (point_c.price - latest_price) / (point_c
                            .price - d_projection)
                completion = max(0.0, min(1.0, completion))
            if completion < self.completion_threshold:
                continue
            pattern = HarmonicPattern(pattern_type=HarmonicPatternType.ABCD,
                direction=PatternDirection.BULLISH if is_bullish else
                PatternDirection.BEARISH, points=points, ratios=ratios,
                completion=completion)
            patterns.append(pattern)
        return patterns

    def _calculate_potential_d_point(self, pattern_type:
        HarmonicPatternType, point_x: PatternPoint, point_a: PatternPoint,
        point_b: PatternPoint, point_c: PatternPoint) ->Optional[PatternPoint]:
        """
        Calculate the potential D point for a pattern, if it exists in the data.
        
        Args:
            pattern_type: Type of harmonic pattern
            point_x: X point
            point_a: A point
            point_b: B point
            point_c: C point
            
        Returns:
            D point if it exists, None otherwise
        """
        return None

    def _project_d_point(self, pattern_type: HarmonicPatternType,
        is_bullish: bool, point_x: PatternPoint, point_a: PatternPoint,
        point_b: PatternPoint, point_c: PatternPoint) ->float:
        """
        Project the price level for the D point based on the pattern type.
        
        Args:
            pattern_type: Type of harmonic pattern
            is_bullish: Whether the pattern is bullish or bearish
            point_x: X point
            point_a: A point
            point_b: B point
            point_c: C point
            
        Returns:
            Projected price level for D point
        """
        ideal_ratios = PATTERN_IDEAL_RATIOS.get(pattern_type, {})
        cd_ratio = ideal_ratios.get('CD', 1.27)
        bc_range = abs(point_c.price - point_b.price)
        if is_bullish:
            return point_c.price + bc_range * cd_ratio
        else:
            return point_c.price - bc_range * cd_ratio

    def _calculate_potential_abcd_point(self, point_a: PatternPoint,
        point_b: PatternPoint, point_c: PatternPoint) ->Optional[PatternPoint]:
        """
        Calculate the potential D point for an ABCD pattern, if it exists in the data.
        
        Args:
            point_a: A point
            point_b: B point
            point_c: C point
            
        Returns:
            D point if it exists, None otherwise
        """
        return None

    def _project_abcd_point(self, is_bullish: bool, point_a: PatternPoint,
        point_b: PatternPoint, point_c: PatternPoint) ->float:
        """
        Project the price level for the D point in an ABCD pattern.
        
        Args:
            is_bullish: Whether the pattern is bullish or bearish
            point_a: A point
            point_b: B point
            point_c: C point
            
        Returns:
            Projected price level for D point
        """
        cd_ratio = PATTERN_IDEAL_RATIOS.get(HarmonicPatternType.ABCD, {}).get(
            'CD', 1.27)
        bc_range = abs(point_c.price - point_b.price)
        if is_bullish:
            return point_c.price + bc_range * cd_ratio
        else:
            return point_c.price - bc_range * cd_ratio

    def _validate_pattern_ratios(self, ratios: Dict[str, float],
        pattern_type: HarmonicPatternType) ->bool:
        """
        Validate pattern ratios against ideal Fibonacci ratios.
        
        Args:
            ratios: Dictionary of calculated ratios
            pattern_type: Type of harmonic pattern
            
        Returns:
            True if ratios are valid, False otherwise
        """
        ideal_ratios = PATTERN_IDEAL_RATIOS.get(pattern_type, {})
        if not ideal_ratios:
            return False
        for ratio_name, actual_ratio in ratios.items():
            if ratio_name not in ideal_ratios:
                continue
            ideal_ratio = ideal_ratios[ratio_name]
            tolerance = ideal_ratios.get(f'{ratio_name}_tolerance', 0.05)
            if abs(actual_ratio - ideal_ratio) > tolerance:
                return False
        return True
