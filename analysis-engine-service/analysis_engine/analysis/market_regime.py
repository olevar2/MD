"""
Market Regime Detection

This module provides functionality to detect and classify market regimes
based on price action, volatility, and trend characteristics.

Note: This is a facade module that re-exports the refactored implementation from the market_regime package.
"""

# Re-export from the refactored package
from analysis_engine.analysis.market_regime.models import (
    RegimeType,
    DirectionType,
    VolatilityLevel,
    RegimeClassification,
    FeatureSet
)
from analysis_engine.analysis.market_regime.detector import RegimeDetector
from analysis_engine.analysis.market_regime.classifier import RegimeClassifier
from analysis_engine.analysis.market_regime.analyzer import MarketRegimeAnalyzer

# For backward compatibility
# Map old names to new ones
MarketRegimeType = RegimeType
VolatilityState = VolatilityLevel
TrendState = DirectionType
MarketRegimeClassifier = RegimeClassifier
MarketRegimeDetector = RegimeDetector

# Define a simple wrapper for backward compatibility
class MarketRegimeResult:
    """Wrapper for backward compatibility with RegimeClassification."""

    def __init__(self, classification):
        self.classification = classification
        self.regime = classification.regime
        self.confidence = classification.confidence
        self.direction = classification.direction
        self.volatility = classification.volatility
        self.timestamp = classification.timestamp
        self.features = classification.features

    @staticmethod
    def from_classification(classification):
        """Create a MarketRegimeResult from a RegimeClassification."""
        return MarketRegimeResult(classification)

# Define a simple wrapper for backward compatibility
class RegimeChangeResult:
    """Wrapper for backward compatibility with regime change events."""

    def __init__(self, new_classification, old_classification=None):
        self.new_regime = new_classification.regime if new_classification else None
        self.old_regime = old_classification.regime if old_classification else None
        self.new_classification = new_classification
        self.old_classification = old_classification
        self.timestamp = new_classification.timestamp if new_classification else None

__all__ = [
    'RegimeType',
    'DirectionType',
    'VolatilityLevel',
    'RegimeClassification',
    'FeatureSet',
    'RegimeDetector',
    'RegimeClassifier',
    'MarketRegimeAnalyzer',
    # Backward compatibility
    'MarketRegimeType',
    'MarketRegimeResult',
    'RegimeChangeResult',
    'VolatilityState',
    'TrendState',
    'MarketRegimeClassifier',
    'MarketRegimeDetector'
]