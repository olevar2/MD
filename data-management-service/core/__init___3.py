"""
Alternative Data Adapters Package.

This package provides adapters for different alternative data sources.
"""

from adapters.base_adapter import BaseAlternativeDataAdapter
from adapters.news_adapter import NewsDataAdapter, MockNewsDataAdapter
from adapters.economic_adapter import EconomicDataAdapter, MockEconomicDataAdapter
from adapters.sentiment_adapter import SentimentDataAdapter, MockSentimentDataAdapter
from adapters.social_media_adapter import SocialMediaDataAdapter, MockSocialMediaDataAdapter
from adapters.adapter_factory_1 import AdapterFactory, MultiSourceAdapterFactory

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
