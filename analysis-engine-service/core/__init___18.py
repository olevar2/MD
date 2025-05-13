"""
Sentiment Models Package

This package provides data models for sentiment analysis results:
- SentimentResult for storing analysis results
- NewsImpact for representing the impact of news on financial markets
"""

from .sentiment_result import SentimentResult
from .news_impact import NewsImpact

__all__ = [
    "SentimentResult",
    "NewsImpact",
]