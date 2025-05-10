"""
Feature Extraction Package.

This package provides feature extractors for alternative data.
"""
from typing import Dict, Type

from data_management_service.alternative.feature_extraction.base_extractor import (
    BaseFeatureExtractor,
    FeatureExtractorRegistry
)
from data_management_service.alternative.feature_extraction.news_extractor import NewsFeatureExtractor
from data_management_service.alternative.feature_extraction.economic_extractor import EconomicFeatureExtractor
from data_management_service.alternative.models import AlternativeDataType

# Create global registry
feature_extractor_registry = FeatureExtractorRegistry()

# Register feature extractors
feature_extractor_registry.register(AlternativeDataType.NEWS, NewsFeatureExtractor)
feature_extractor_registry.register(AlternativeDataType.ECONOMIC, EconomicFeatureExtractor)

__all__ = [
    "BaseFeatureExtractor",
    "FeatureExtractorRegistry",
    "NewsFeatureExtractor",
    "EconomicFeatureExtractor",
    "feature_extractor_registry"
]
