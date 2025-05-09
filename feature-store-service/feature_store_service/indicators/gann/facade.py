"""
Gann Tools Facade Module.

This module provides a facade for backward compatibility with the original gann_tools.py module.
"""

# Re-export all classes from the new module structure
from feature_store_service.indicators.gann.angles import GannAngles as NewGannAngles
from feature_store_service.indicators.gann.fans import GannFan as NewGannFan
from feature_store_service.indicators.gann.squares.square_of_9 import GannSquare
from feature_store_service.indicators.gann.time_cycles import GannTimeCycles
from feature_store_service.indicators.gann.grid import GannGrid
from feature_store_service.indicators.gann.box import GannBox
from feature_store_service.indicators.gann.squares.square_of_144 import GannSquare144
from feature_store_service.indicators.gann.hexagon import GannHexagon

# Import legacy adapters for backward compatibility
from feature_store_service.indicators.gann.legacy_adapters import (
    GannAngles,
    GannSquare9,
    GannFan
)

# For backward compatibility, we can add any additional functions or classes here
# that were in the original gann_tools.py but are not part of the new structure.

# Export all classes
__all__ = [
    'GannAngles',
    'GannFan',
    'GannSquare',
    'GannTimeCycles',
    'GannGrid',
    'GannBox',
    'GannSquare144',
    'GannHexagon',
    'GannSquare9',
    'NewGannAngles',
    'NewGannFan'
]
