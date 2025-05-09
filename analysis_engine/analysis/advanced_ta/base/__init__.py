"""
Base classes for advanced technical analysis.
"""

from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence level for pattern detection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MarketDirection(Enum):
    """Market direction."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


class PatternResult:
    """Base class for pattern detection results."""
    
    def __init__(self, **kwargs):
        """Initialize pattern result."""
        self.pattern_name = kwargs.get("pattern_name", "")
        self.pattern_type = kwargs.get("pattern_type", None)
        self.confidence = kwargs.get("confidence", ConfidenceLevel.LOW)
        self.direction = kwargs.get("direction", MarketDirection.SIDEWAYS)
        self.start_time = kwargs.get("start_time", None)
        self.end_time = kwargs.get("end_time", None)
        self.start_price = kwargs.get("start_price", 0.0)
        self.end_price = kwargs.get("end_price", 0.0)
    
    def to_dict(self):
        """Convert to dictionary representation."""
        return {
            "pattern_name": self.pattern_name,
            "pattern_type": self.pattern_type.value if self.pattern_type else None,
            "confidence": self.confidence.value,
            "direction": self.direction.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "start_price": self.start_price,
            "end_price": self.end_price
        }


class PatternRecognitionBase:
    """Base class for pattern recognition."""
    
    def __init__(self, name, parameters=None):
        """Initialize pattern recognition."""
        self.name = name
        self.parameters = parameters or {}
    
    def find_patterns(self, df):
        """Find patterns in data."""
        raise NotImplementedError("Subclasses must implement find_patterns")
    
    def calculate(self, df):
        """Calculate indicator values."""
        raise NotImplementedError("Subclasses must implement calculate")


class AdvancedAnalysisBase:
    """Base class for advanced analysis."""
    
    def __init__(self, name, parameters=None):
        """Initialize advanced analysis."""
        self.name = name
        self.parameters = parameters or {}
    
    def calculate(self, df):
        """Calculate analysis values."""
        raise NotImplementedError("Subclasses must implement calculate")