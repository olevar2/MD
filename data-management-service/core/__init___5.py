"""
Feature Extraction Package.

This package provides feature extractors for alternative data.
"""
from typing import Dict, Type

from core.base_extractor import (
    BaseFeatureExtractor,
    FeatureExtractorRegistry
)
from core.news_extractor import NewsFeatureExtractor
from core.economic_extractor import EconomicFeatureExtractor
from models.models import AlternativeDataType

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
