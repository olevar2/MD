"""
Backward Compatibility Module for News Analyzer

This module provides backward compatibility for code that uses the original
NewsAnalyzer class from the nlp package.
"""

import warnings
from typing import Dict, List, Any

from analysis_engine.analysis.sentiment.analyzers.nlp_analyzer import NLPSentimentAnalyzer
from analysis_engine.models.analysis_result import AnalysisResult


class NewsAnalyzer(NLPSentimentAnalyzer):
    """
    Backward compatibility class for the original NewsAnalyzer.
    
    This class inherits from the new NLPSentimentAnalyzer and provides
    the same interface as the original NewsAnalyzer class.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the backward compatibility class.
        
        Args:
            parameters: Configuration parameters for the analyzer
        """
        warnings.warn(
            "NewsAnalyzer from nlp package is deprecated and will be removed in a future version. "
            "Use NLPSentimentAnalyzer from sentiment package instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(parameters)
        self.name = "news_analyzer"  # Override name for backward compatibility
    
    def categorize_news(self, text: str) -> Dict[str, float]:
        """
        Backward compatibility method for categorize_news.
        
        Args:
            text: Text to categorize
            
        Returns:
            Dictionary mapping categories to confidence scores
        """
        return self.categorize_content(text)
    
    def assess_pair_impact(self, text: str, categories: Dict[str, float], 
                         entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Backward compatibility method for assess_pair_impact.
        
        Args:
            text: Text to analyze
            categories: Content categories with scores
            entities: Extracted entities
            
        Returns:
            Dictionary mapping currency pairs to impact scores and directions
        """
        return self.assess_market_impact(text, categories, entities)