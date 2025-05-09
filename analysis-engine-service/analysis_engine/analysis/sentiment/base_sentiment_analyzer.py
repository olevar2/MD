"""
Base Sentiment Analyzer

This module provides the base class for all sentiment analyzers.
"""

from typing import Dict, List, Any, Union
import logging
from abc import abstractmethod

from analysis_engine.analysis.base_analyzer import BaseAnalyzer
from analysis_engine.models.analysis_result import AnalysisResult

logger = logging.getLogger(__name__)


class BaseSentimentAnalyzer(BaseAnalyzer):
    """
    Base class for all sentiment analyzers.
    
    This class defines the common interface and functionality for all sentiment
    analyzers, regardless of the specific technique used.
    """
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        """
        Initialize the base sentiment analyzer
        
        Args:
            name: Name identifier for the analyzer
            parameters: Configuration parameters for the analyzer
        """
        super().__init__(name, parameters)
    
    @abstractmethod
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        pass
    
    @abstractmethod
    def categorize_content(self, text: str) -> Dict[str, float]:
        """
        Categorize content into predefined categories
        
        Args:
            text: Text to categorize
            
        Returns:
            Dictionary mapping categories to confidence scores
        """
        pass
    
    @abstractmethod
    def assess_market_impact(self, text: str, categories: Dict[str, float], 
                           entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Assess potential market impact
        
        Args:
            text: Text to analyze
            categories: Content categories with scores
            entities: Extracted entities
            
        Returns:
            Dictionary mapping market instruments to impact scores and directions
        """
        pass
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of extracted entities with metadata
        """
        # Default implementation - should be overridden by subclasses
        return []
    
    def filter_by_recency(self, items: List[Dict[str, Any]], 
                        timestamp_field: str = "timestamp",
                        lookback_hours: int = 24) -> List[Dict[str, Any]]:
        """
        Filter items by recency
        
        Args:
            items: List of items to filter
            timestamp_field: Field name containing the timestamp
            lookback_hours: Hours to look back
            
        Returns:
            Filtered list of items
        """
        from datetime import datetime, timedelta
        
        filtered_items = []
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        
        for item in items:
            if timestamp_field in item:
                # Parse timestamp (assuming ISO format)
                try:
                    item_time = datetime.fromisoformat(item[timestamp_field].replace('Z', '+00:00'))
                    if item_time >= cutoff_time:
                        filtered_items.append(item)
                except (ValueError, TypeError):
                    # If timestamp parsing fails, include the item anyway
                    filtered_items.append(item)
            else:
                filtered_items.append(item)
                
        return filtered_items