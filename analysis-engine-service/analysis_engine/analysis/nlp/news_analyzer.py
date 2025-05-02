"""
News Analyzer Module

This module provides functionality for analyzing financial news
and extracting relevant information for trading decisions.
"""

from typing import Dict, List, Any, Union
import logging
import re
from datetime import datetime, timedelta
import json

from analysis_engine.analysis.nlp.base_nlp_analyzer import BaseNLPAnalyzer
from analysis_engine.models.analysis_result import AnalysisResult

logger = logging.getLogger(__name__)

class NewsAnalyzer(BaseNLPAnalyzer):
    """
    Analyzer for financial news content with market impact assessment.
    
    This analyzer processes financial news articles, extracts relevant entities,
    analyzes sentiment, and assesses potential market impact on forex pairs.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the news analyzer
        
        Args:
            parameters: Configuration parameters for the analyzer
        """
        default_params = {
            "spacy_model": "en_core_web_sm",
            "impact_threshold": 0.6,  # Threshold for considering news as impactful
            "lookback_hours": 24,  # Hours of news to analyze
            "currency_sensitivity_map": {  # Default currency sensitivities to news topics
                "USD": ["federal_reserve", "interest_rates", "inflation", "employment", "gdp", "fomc", "trade_balance"],
                "EUR": ["ecb", "eurozone", "euro", "european_union", "draghi", "lagarde", "france", "germany"],
                "GBP": ["boe", "brexit", "uk", "bank_of_england", "trade_deal"],
                "JPY": ["boj", "japan", "kuroda", "abenomics", "safe_haven"],
                "AUD": ["rba", "australia", "commodities", "china_trade"],
                "NZD": ["rbnz", "new_zealand", "dairy"],
                "CAD": ["boc", "canada", "oil", "natural_resources"],
                "CHF": ["snb", "swiss", "safe_haven", "swiss_national_bank"]
            }
        }
        
        # Merge default parameters with provided parameters
        merged_params = {**default_params, **(parameters or {})}
        super().__init__("news_analyzer", merged_params)
        
        # Initialize news category classifiers
        self._initialize_news_classifiers()
        
    def _initialize_news_classifiers(self):
        """Initialize the news category classifiers"""
        # Keyword-based classifiers for news categories
        self.category_keywords = {
            "monetary_policy": ["interest rate", "central bank", "monetary policy", "rate decision", "rate hike", 
                              "rate cut", "policy meeting", "fomc", "committee", "basis points", "hawkish", "dovish"],
                              
            "economic_indicators": ["gdp", "inflation", "cpi", "ppi", "employment", "unemployment", "nonfarm", "payroll", 
                                   "jobs", "retail sales", "industrial production", "manufacturing", "pmi", "ism", 
                                   "trade balance", "deficit", "surplus"],
                                   
            "geopolitical": ["war", "conflict", "sanctions", "tariff", "trade war", "election", "vote", "referendum", 
                            "brexit", "political", "government", "shutdown"],
                            
            "company_news": ["earnings", "profit", "revenue", "guidance", "forecast", "outlook", "acquisition", 
                            "merger", "takeover", "ipo", "stock", "share"],
                            
            "market_sentiment": ["risk", "rally", "sell-off", "correction", "bull", "bear", "sentiment", "mood", 
                               "fear", "greed", "panic", "optimism", "pessimism", "volatility"]
        }
    
    def categorize_news(self, text: str) -> Dict[str, float]:
        """
        Categorize news text into predefined categories
        
        Args:
            text: News text
            
        Returns:
            Dictionary mapping categories to confidence scores
        """
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.category_keywords.items():
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            score = min(matches / (len(keywords) * 0.3), 1.0)  # Normalize score
            scores[category] = score
            
        return scores
        
    def assess_pair_impact(self, text: str, categories: Dict[str, float], entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Assess potential impact on currency pairs
        
        Args:
            text: News text
            categories: News categories with scores
            entities: Extracted entities
            
        Returns:
            Dictionary mapping currency pairs to impact scores and directions
        """
        # Extract relevant currencies from text
        mentioned_currencies = set()
        currency_sensitivity_map = self.parameters["currency_sensitivity_map"]
        
        # Check for explicit currency mentions
        text_lower = text.lower()
        for currency, topics in currency_sensitivity_map.items():
            if currency.lower() in text_lower or any(topic.lower() in text_lower for topic in topics):
                mentioned_currencies.add(currency)
                
        # Add currencies from extracted entities
        for entity in entities:
            if entity["label"] == "MONEY" or entity["label"] == "ORG":
                for currency in currency_sensitivity_map.keys():
                    if currency.lower() in entity["text"].lower():
                        mentioned_currencies.add(currency)
                        
        # If no currencies detected, use major currencies as default
        if not mentioned_currencies:
            mentioned_currencies = {"USD", "EUR"}
            
        # Calculate sentiment
        sentiment = self.analyze_sentiment(text)
        
        # Generate potential impacts for currency pairs
        pair_impacts = {}
        
        # Major currency pairs with base currency
        major_pairs = [
            ("EUR", "USD"), ("USD", "JPY"), ("GBP", "USD"), ("USD", "CHF"), 
            ("USD", "CAD"), ("AUD", "USD"), ("NZD", "USD")
        ]
        
        # Assess impact on currency pairs
        for base, quote in major_pairs:
            # Check if either currency in the pair is mentioned
            if base in mentioned_currencies or quote in mentioned_currencies:
                # Determine direction of impact
                direction = 0.0  # Neutral by default
                
                # If both currencies mentioned, compare relative impact
                if base in mentioned_currencies and quote in mentioned_currencies:
                    # Use sentiment to determine which currency might be more affected
                    # This is a simplified approach and would need refinement in production
                    direction = 0.2 * sentiment["compound"]  # Dampened impact when both affected
                elif base in mentioned_currencies:
                    direction = sentiment["compound"]  # Base currency affected
                elif quote in mentioned_currencies:
                    direction = -sentiment["compound"]  # Quote currency affected (inverted)
                
                # Calculate impact magnitude
                magnitude = max(categories.values()) * abs(sentiment["compound"])
                
                # Store result
                pair_name = f"{base}/{quote}"
                pair_impacts[pair_name] = {
                    "impact_score": magnitude,
                    "direction": direction,
                    "sentiment": sentiment["compound"],
                    "categories": {k: v for k, v in categories.items() if v > 0.2}
                }
                
        return pair_impacts
    
    def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        Analyze news data
        
        Args:
            data: Dictionary containing news articles with text and metadata
            
        Returns:
            AnalysisResult containing analysis results
        """
        if not data or "news_items" not in data:
            return AnalysisResult(
                analyzer_name=self.name,
                result_data={"error": "No news data provided"},
                is_valid=False
            )
            
        news_items = data["news_items"]
        if not isinstance(news_items, list) or not news_items:
            return AnalysisResult(
                analyzer_name=self.name,
                result_data={"error": "Invalid news data format or empty news list"},
                is_valid=False
            )
            
        # Filter news by recency if timestamp available
        filtered_news = []
        lookback_hours = self.parameters["lookback_hours"]
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        
        for news in news_items:
            if "timestamp" in news:
                # Parse timestamp (assuming ISO format)
                try:
                    news_time = datetime.fromisoformat(news["timestamp"].replace('Z', '+00:00'))
                    if news_time >= cutoff_time:
                        filtered_news.append(news)
                except (ValueError, TypeError):
                    # If timestamp parsing fails, include the news anyway
                    filtered_news.append(news)
            else:
                filtered_news.append(news)
                
        if not filtered_news:
            return AnalysisResult(
                analyzer_name=self.name,
                result_data={"message": "No recent news found"},
                is_valid=True
            )
            
        # Analyze each news item
        results = []
        for news in filtered_news:
            text = news.get("title", "") + " " + news.get("content", "")
            
            # Skip empty news
            if not text.strip():
                continue
                
            # Extract entities
            entities = self.extract_entities(text)
            
            # Categorize news
            categories = self.categorize_news(text)
            
            # Assess potential impact on currency pairs
            pair_impacts = self.assess_pair_impact(text, categories, entities)
            
            # Calculate overall impact score
            max_impact = max([impact["impact_score"] for impact in pair_impacts.values()]) if pair_impacts else 0
            
            news_result = {
                "id": news.get("id", ""),
                "title": news.get("title", ""),
                "source": news.get("source", ""),
                "timestamp": news.get("timestamp", ""),
                "sentiment": self.analyze_sentiment(text),
                "categories": categories,
                "entities": entities,
                "pair_impacts": pair_impacts,
                "overall_impact_score": max_impact
            }
            
            results.append(news_result)
            
        # Sort results by impact score
        results.sort(key=lambda x: x["overall_impact_score"], reverse=True)
        
        # Identify high-impact news
        impact_threshold = self.parameters["impact_threshold"]
        high_impact_news = [news for news in results if news["overall_impact_score"] >= impact_threshold]
        
        # Summary by currency pair
        pair_summary = {}
        for news in results:
            for pair, impact in news["pair_impacts"].items():
                if pair not in pair_summary:
                    pair_summary[pair] = {
                        "count": 0,
                        "avg_impact": 0,
                        "avg_direction": 0,
                        "high_impact_count": 0
                    }
                
                pair_summary[pair]["count"] += 1
                pair_summary[pair]["avg_impact"] += impact["impact_score"]
                pair_summary[pair]["avg_direction"] += impact["direction"]
                
                if news["overall_impact_score"] >= impact_threshold:
                    pair_summary[pair]["high_impact_count"] += 1
                    
        # Calculate averages
        for pair in pair_summary:
            count = pair_summary[pair]["count"]
            if count > 0:
                pair_summary[pair]["avg_impact"] /= count
                pair_summary[pair]["avg_direction"] /= count
        
        return AnalysisResult(
            analyzer_name=self.name,
            result_data={
                "news_count": len(results),
                "high_impact_count": len(high_impact_news),
                "news_items": results,
                "high_impact_news": high_impact_news,
                "pair_summary": pair_summary
            },
            is_valid=True
        )
