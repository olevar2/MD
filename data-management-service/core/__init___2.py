"""
Alternative Data Integration Framework.

This package provides a framework for integrating alternative data sources
into the forex trading platform.
"""

from models.models import (
    AlternativeDataType,
    DataFrequency,
    DataReliability,
    DataSourceMetadata,
    AlternativeDataSource,
    DataValidationRule,
    DataTransformationRule,
    AlternativeDataSchema,
    CorrelationMetric,
    AlternativeDataCorrelation,
    FeatureExtractionConfig
)
from services.service import AlternativeDataService
from adapters.adapter_factory_1 import AdapterFactory, MultiSourceAdapterFactory
from data_management_service.alternative.feature_extraction import (
    BaseFeatureExtractor,
    NewsFeatureExtractor,
    EconomicFeatureExtractor,
    feature_extractor_registry
)
from data_management_service.alternative.trading import (
    BaseTradingSignalGenerator,
    NewsTradingSignalGenerator,
    EconomicTradingSignalGenerator
)
from data_management_service.alternative.correlation import (
    BaseCorrelationAnalyzer,
    StandardCorrelationAnalyzer
)

__all__ = [
    # Models
    "AlternativeDataType",
    "DataFrequency",
    "DataReliability",
    "DataSourceMetadata",
    "AlternativeDataSource",
    "DataValidationRule",
    "DataTransformationRule",
    "AlternativeDataSchema",
    "CorrelationMetric",
    "AlternativeDataCorrelation",
    "FeatureExtractionConfig",

    # Service
    "AlternativeDataService",

    # Adapters
    "AdapterFactory",
    "MultiSourceAdapterFactory",

    # Feature Extraction
    "BaseFeatureExtractor",
    "NewsFeatureExtractor",
    "EconomicFeatureExtractor",
    "feature_extractor_registry",

    # Trading
    "BaseTradingSignalGenerator",
    "NewsTradingSignalGenerator",
    "EconomicTradingSignalGenerator",

    # Correlation
    "BaseCorrelationAnalyzer",
    "StandardCorrelationAnalyzer"
]
