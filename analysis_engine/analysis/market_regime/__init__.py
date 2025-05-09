"""
Market Regime Analysis Package

This package provides tools for analyzing and classifying market regimes.
It includes components for feature extraction, regime detection, and classification.

Public API:
    MarketRegimeAnalyzer: Main class for performing market regime analysis
    RegimeType: Enum representing different market regime types
    RegimeClassification: Data model for regime classification results
"""

from analysis_engine.analysis.market_regime.analyzer import MarketRegimeAnalyzer
from analysis_engine.analysis.market_regime.models import RegimeType, RegimeClassification

__all__ = ['MarketRegimeAnalyzer', 'RegimeType', 'RegimeClassification']