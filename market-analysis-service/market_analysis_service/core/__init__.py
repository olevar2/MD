"""
Core modules for Market Analysis Service.
"""

from market_analysis_service.core.pattern_recognition import PatternRecognizer
from market_analysis_service.core.support_resistance import SupportResistanceDetector
from market_analysis_service.core.market_regime import MarketRegimeDetector
from market_analysis_service.core.correlation_analysis import CorrelationAnalyzer
from market_analysis_service.core.volatility_analysis import VolatilityAnalyzer
from market_analysis_service.core.sentiment_analysis import SentimentAnalyzer
from market_analysis_service.core.service_dependencies import (
    get_data_pipeline_adapter,
    get_analysis_coordinator_adapter,
    get_feature_store_adapter,
    get_analysis_repository,
    get_market_analysis_service
)

__all__ = [
    'PatternRecognizer',
    'SupportResistanceDetector',
    'MarketRegimeDetector',
    'CorrelationAnalyzer',
    'VolatilityAnalyzer',
    'SentimentAnalyzer',
    'get_data_pipeline_adapter',
    'get_analysis_coordinator_adapter',
    'get_feature_store_adapter',
    'get_analysis_repository',
    'get_market_analysis_service'
]