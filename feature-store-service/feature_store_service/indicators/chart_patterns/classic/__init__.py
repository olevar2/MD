"""
Classic Chart Patterns Package.

This package provides implementations of classic chart patterns like Head and Shoulders,
Double Tops/Bottoms, Triangle patterns, etc.
"""

from feature_store_service.indicators.chart_patterns.classic.head_and_shoulders import HeadAndShouldersPattern
from feature_store_service.indicators.chart_patterns.classic.double_formations import DoubleFormationPattern
from feature_store_service.indicators.chart_patterns.classic.triple_formations import TripleFormationPattern
from feature_store_service.indicators.chart_patterns.classic.triangles import TrianglePattern
from feature_store_service.indicators.chart_patterns.classic.flags_pennants import FlagPennantPattern
from feature_store_service.indicators.chart_patterns.classic.wedges import WedgePattern
from feature_store_service.indicators.chart_patterns.classic.rectangles import RectanglePattern

__all__ = [
    'HeadAndShouldersPattern',
    'DoubleFormationPattern',
    'TripleFormationPattern',
    'TrianglePattern',
    'FlagPennantPattern',
    'WedgePattern',
    'RectanglePattern'
]
