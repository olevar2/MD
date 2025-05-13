"""
Sentiment Result Model

This module provides data models for sentiment analysis results.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SentimentResult:
    """
    Data model for sentiment analysis results.
    
    This class represents the result of sentiment analysis on a piece of content,
    including sentiment scores, categories, and market impact.
    """
    
    # Basic metadata
    id: str = ""
    title: str = ""
    source: str = ""
    timestamp: str = ""
    content_snippet: str = ""
    
    # Sentiment scores
    compound_score: float = 0.0
    positive_score: float = 0.0
    negative_score: float = 0.0
    neutral_score: float = 0.0
    
    # Categories and entities
    categories: Dict[str, float] = field(default_factory=dict)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    
    # Market impact
    market_impacts: Dict[str, Dict[str, float]] = field(default_factory=dict)
    overall_impact_score: float = 0.0
    
    @property
    def sentiment_label(self) -> str:
        """Get the sentiment label based on the compound score."""
        if self.compound_score >= 0.05:
            return "positive"
        elif self.compound_score <= -0.05:
            return "negative"
        else:
            return "neutral"
    
    @property
    def top_category(self) -> str:
        """Get the top category based on confidence score."""
        if not self.categories:
            return "unknown"
        
        return max(self.categories.items(), key=lambda x: x[1])[0]
    
    @property
    def top_impacted_instrument(self) -> str:
        """Get the top impacted instrument based on impact score."""
        if not self.market_impacts:
            return "none"
        
        return max(self.market_impacts.items(), 
                  key=lambda x: abs(x[1].get("impact_score", 0)))[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "timestamp": self.timestamp,
            "content_snippet": self.content_snippet,
            "sentiment": {
                "compound": self.compound_score,
                "positive": self.positive_score,
                "negative": self.negative_score,
                "neutral": self.neutral_score,
                "label": self.sentiment_label
            },
            "categories": self.categories,
            "entities": self.entities,
            "market_impacts": self.market_impacts,
            "overall_impact_score": self.overall_impact_score,
            "top_category": self.top_category,
            "top_impacted_instrument": self.top_impacted_instrument
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SentimentResult':
        """Create from dictionary representation."""
        sentiment = data.get("sentiment", {})
        
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            source=data.get("source", ""),
            timestamp=data.get("timestamp", ""),
            content_snippet=data.get("content_snippet", ""),
            compound_score=sentiment.get("compound", 0.0),
            positive_score=sentiment.get("positive", 0.0),
            negative_score=sentiment.get("negative", 0.0),
            neutral_score=sentiment.get("neutral", 0.0),
            categories=data.get("categories", {}),
            entities=data.get("entities", []),
            market_impacts=data.get("market_impacts", {}),
            overall_impact_score=data.get("overall_impact_score", 0.0)
        )