"""
Base indicator class.
"""


class BaseIndicator:
    """Base class for all indicators."""
    
    category = "generic"
    
    def __init__(self, **kwargs):
        """Initialize base indicator."""
        self.name = kwargs.get("name", "indicator")
    
    def calculate(self, data):
        """Calculate indicator values."""
        raise NotImplementedError("Subclasses must implement calculate")