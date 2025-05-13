"""
Advanced Pattern Recognition Package.

This package provides advanced pattern recognition capabilities for forex market analysis.
"""

from core.base import (
    AdvancedPatternType,
    AdvancedPatternRecognizer,
    PatternDirection,
    PatternStrength
)

# Import pattern recognizers
from feature_store_service.indicators.advanced_patterns.renko import RenkoPatternRecognizer
from feature_store_service.indicators.advanced_patterns.ichimoku import IchimokuPatternRecognizer
from feature_store_service.indicators.advanced_patterns.wyckoff import WyckoffPatternRecognizer
from feature_store_service.indicators.advanced_patterns.heikin_ashi import HeikinAshiPatternRecognizer
from feature_store_service.indicators.advanced_patterns.vsa import VSAPatternRecognizer
from feature_store_service.indicators.advanced_patterns.market_profile import MarketProfileAnalyzer
from feature_store_service.indicators.advanced_patterns.point_and_figure import PointAndFigureAnalyzer
from feature_store_service.indicators.advanced_patterns.wolfe_wave import WolfeWaveDetector
from feature_store_service.indicators.advanced_patterns.pitchfork import PitchforkAnalyzer
from feature_store_service.indicators.advanced_patterns.divergence import DivergenceDetector

# Import facade for backward compatibility
from core.facade import AdvancedPatternFacade

__all__ = [
    "AdvancedPatternType",
    "AdvancedPatternRecognizer",
    "PatternDirection",
    "PatternStrength",
    "RenkoPatternRecognizer",
    "IchimokuPatternRecognizer",
    "WyckoffPatternRecognizer",
    "HeikinAshiPatternRecognizer",
    "VSAPatternRecognizer",
    "MarketProfileAnalyzer",
    "PointAndFigureAnalyzer",
    "WolfeWaveDetector",
    "PitchforkAnalyzer",
    "DivergenceDetector",
    "AdvancedPatternFacade"
]