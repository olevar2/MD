"""
Correlation Analysis Module for Alternative Data.

This module provides functionality for analyzing correlations between alternative data and market movements.
"""

from data_management_service.alternative.correlation.analyzer import (
    BaseCorrelationAnalyzer,
    StandardCorrelationAnalyzer
)

__all__ = [
    "BaseCorrelationAnalyzer",
    "StandardCorrelationAnalyzer"
]
