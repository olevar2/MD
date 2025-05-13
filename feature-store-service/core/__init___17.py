"""
Classic Chart Patterns Package.

This package provides implementations of classic chart patterns like Head and Shoulders,
Double Tops/Bottoms, Triangle patterns, etc.
"""

from core.head_and_shoulders import HeadAndShouldersPattern
from core.double_formations import DoubleFormationPattern
from core.triple_formations import TripleFormationPattern
from core.triangles import TrianglePattern
from core.flags_pennants import FlagPennantPattern
from core.wedges import WedgePattern
from core.rectangles import RectanglePattern

__all__ = [
    'HeadAndShouldersPattern',
    'DoubleFormationPattern',
    'TripleFormationPattern',
    'TrianglePattern',
    'FlagPennantPattern',
    'WedgePattern',
    'RectanglePattern'
]
