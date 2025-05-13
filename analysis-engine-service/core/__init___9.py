"""
Advanced Technical Analysis Package

This package provides comprehensive implementations of advanced technical analysis methods
including Gann, Fibonacci, Elliott Wave, Pivot Points, Fractals, Harmonic Patterns,
Classic Patterns, Candlestick Patterns, Volume/Volatility Analysis, Time Cycle Analysis,
Correlation/Divergence Analysis, Multi-Timeframe Analysis, and Confluence Analysis.

Each module provides both standard calculation and incremental update capabilities for
optimal performance in real-time trading systems.
"""

from analysis_engine.analysis.advanced_ta.base import (
    AdvancedAnalysisBase,
    PatternRecognitionBase,
    PatternResult,
    ConfidenceLevel,
    MarketDirection,
    AnalysisTimeframe
)

# Import Phase 4: Advanced Volatility Indicators
from analysis_engine.analysis.advanced_ta.advanced_volatility import (
    DonchianChannels,
    PriceEnvelopes,
    VIXFix,
    HistoricalVolatility
)

# Import Phase 4: Advanced Price Indicators
from analysis_engine.analysis.advanced_ta.advanced_price_charts import (
    HeikinAshi,
    RenkoCharts,
    PointAndFigure
)

# Import Phase 4: Statistical and Regression Indicators
from analysis_engine.analysis.advanced_ta.statistical_regression import (
    StandardDeviationAnalyzer,
    LinearRegressionAnalyzer,
    LinearRegressionChannelAnalyzer
)

# Version and metadata
__version__ = '0.1.0'
