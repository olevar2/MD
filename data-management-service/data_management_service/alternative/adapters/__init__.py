"""
Alternative Data Adapters Package.

This package provides adapters for different alternative data sources.
"""

from data_management_service.alternative.adapters.base_adapter import BaseAlternativeDataAdapter
from data_management_service.alternative.adapters.news_adapter import NewsDataAdapter, MockNewsDataAdapter
from data_management_service.alternative.adapters.economic_adapter import EconomicDataAdapter, MockEconomicDataAdapter
from data_management_service.alternative.adapters.sentiment_adapter import SentimentDataAdapter, MockSentimentDataAdapter
from data_management_service.alternative.adapters.social_media_adapter import SocialMediaDataAdapter, MockSocialMediaDataAdapter
from data_management_service.alternative.adapters.adapter_factory import AdapterFactory, MultiSourceAdapterFactory

__all__ = [
    "BaseAlternativeDataAdapter",
    "NewsDataAdapter",
    "MockNewsDataAdapter",
    "EconomicDataAdapter",
    "MockEconomicDataAdapter",
    "SentimentDataAdapter",
    "MockSentimentDataAdapter",
    "SocialMediaDataAdapter",
    "MockSocialMediaDataAdapter",
    "AdapterFactory",
    "MultiSourceAdapterFactory"
]
