"""
Technical Indicators Package.

This package provides various technical indicators for forex market analysis.
"""

from feature_store_service.indicators.base_indicator import BaseIndicator
from feature_store_service.indicators.moving_averages import (
    SimpleMovingAverage,
    ExponentialMovingAverage,
    WeightedMovingAverage,
)
from feature_store_service.indicators.oscillators import (
    RelativeStrengthIndex,
    Stochastic,
    MACD,
)
from feature_store_service.indicators.volatility import (
    BollingerBands,
    AverageTrueRange,
    KeltnerChannels,
    # Phase 4: Advanced Volatility Indicators
    DonchianChannels,
    PriceEnvelopes,
    VIXFixIndicator as VIXFix,
    HistoricalVolatility,
)
from feature_store_service.indicators.advanced_price_indicators import (
    # Phase 4: Advanced Price Indicators
    IchimokuCloud,
    HeikinAshi,
    RenkoCharts,
    PointAndFigure,
)
from feature_store_service.indicators.statistical_regression_indicators import (
    # Phase 4: Statistical and Regression Indicators
    StandardDeviationIndicator,
    LinearRegressionIndicator,
    LinearRegressionChannel,
)
from feature_store_service.indicators.advanced_patterns import (
    # Advanced Pattern Recognition
    AdvancedPatternFacade,
    RenkoPatternRecognizer,
    IchimokuPatternRecognizer,
    WyckoffPatternRecognizer,
    HeikinAshiPatternRecognizer,
    VSAPatternRecognizer,
    MarketProfileAnalyzer,
    PointAndFigureAnalyzer,
    WolfeWaveDetector,
    PitchforkAnalyzer,
    DivergenceDetector,
)

__all__ = [
    "BaseIndicator",
    "SimpleMovingAverage",
    "ExponentialMovingAverage",
    "WeightedMovingAverage",
    "RelativeStrengthIndex",
    "Stochastic",
    "MACD",
    "BollingerBands",
    "AverageTrueRange",
    "KeltnerChannels",
    # Phase 4: Advanced Volatility Indicators
    "DonchianChannels",
    "PriceEnvelopes",
    "VIXFix",
    "HistoricalVolatility",
    # Phase 4: Advanced Price Indicators
    "IchimokuCloud",
    "HeikinAshi",
    "RenkoCharts",
    "PointAndFigure",
    # Phase 4: Statistical and Regression Indicators
    "StandardDeviationIndicator",
    "LinearRegressionIndicator",
    "LinearRegressionChannel",
    # Advanced Pattern Recognition
    "AdvancedPatternFacade",
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
]