"""
Gann Tools Facade Module.

This module provides a facade for backward compatibility with the original gann_tools.py module.
"""

# Re-export all classes from the new module structure
from core.angles import GannAngles as NewGannAngles
from core.fans_1 import GannFan as NewGannFan
from core.square_of_9 import GannSquare
from core.time_cycles import GannTimeCycles
from core.grid import GannGrid
from core.box import GannBox
from core.square_of_144 import GannSquare144
from core.hexagon import GannHexagon

# Import legacy adapters for backward compatibility
from adapters.legacy_adapters import (
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
