"""
NLP-based Sentiment Analyzer

This module provides sentiment analysis using natural language processing techniques.
"""

from typing import Dict, List, Any, Union
import logging
import re
from datetime import datetime, timedelta

from analysis_engine.analysis.sentiment.base_sentiment_analyzer import BaseSentimentAnalyzer
from analysis_engine.analysis.sentiment.models.sentiment_result import SentimentResult
from analysis_engine.models.analysis_result import AnalysisResult

logger = logging.getLogger(__name__)


class NLPSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Sentiment analyzer using natural language processing techniques.
    
    This analyzer uses NLP to process text, extract entities, analyze sentiment,
    and assess potential market impact.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the NLP sentiment analyzer
        
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
        super().__init__("nlp_sentiment_analyzer", merged_params)
        
        # Initialize NLP pipeline
        self._initialize_nlp_pipeline()
        
        # Initialize news category classifiers
        self._initialize_category_classifiers()
    
    def _initialize_nlp_pipeline(self):
        """Initialize the NLP pipeline"""
        try:
            import spacy
            self.nlp = spacy.load(self.parameters["spacy_model"])
            logger.info(f"Loaded spaCy model: {self.parameters['spacy_model']}")
        except Exception as e:
            logger.warning(f"Failed to load spaCy model: {str(e)}")
            self.nlp = None
    
    def _initialize_category_classifiers(self):
        """Initialize the category classifiers"""
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
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        # Ensure NLP pipeline is initialized
        if not self.nlp:
            self._initialize_nlp_pipeline()
            if not self.nlp:
                # Fallback to basic implementation if NLP pipeline initialization fails
                return self._basic_sentiment_analysis(text)
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Count positive and negative words using a simple lexicon approach
            positive_words = set(['increase', 'growth', 'positive', 'bullish', 'upward', 'gain', 'profit',
                                'strengthen', 'improved', 'recovery', 'strong', 'robust', 'exceed'])
            negative_words = set(['decrease', 'decline', 'negative', 'bearish', 'downward', 'loss', 'deficit',
                                'weaken', 'deteriorate', 'recession', 'weak', 'below', 'fail'])
            
            # Count occurrences
            positive_count = sum(1 for token in doc if token.text.lower() in positive_words)
            negative_count = sum(1 for token in doc if token.text.lower() in negative_words)
            total_count = len(doc)
            
            # Calculate scores
            if total_count > 0:
                positive_score = positive_count / total_count
                negative_score = negative_count / total_count
                neutral_score = 1.0 - (positive_score + negative_score)
                
                # Compound score: -1 (very negative) to 1 (very positive)
                compound_score = (positive_score - negative_score) * 2
            else:
                positive_score = 0.0
                negative_score = 0.0
                neutral_score = 1.0
                compound_score = 0.0
            
            return {
                "compound": compound_score,
                "positive": positive_score,
                "negative": negative_score,
                "neutral": neutral_score
            }
        
        except Exception as e:
            logger.warning(f"Error in sentiment analysis: {str(e)}")
            return self._basic_sentiment_analysis(text)
    
    def _basic_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Basic sentiment analysis fallback
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        # Simple word counting approach
        text_lower = text.lower()
        
        # Basic positive and negative word lists
        positive_words = ['increase', 'growth', 'positive', 'bullish', 'upward', 'gain', 'profit',
                         'strengthen', 'improved', 'recovery', 'strong', 'robust', 'exceed']
        negative_words = ['decrease', 'decline', 'negative', 'bearish', 'downward', 'loss', 'deficit',
                         'weaken', 'deteriorate', 'recession', 'weak', 'below', 'fail']
        
        # Count occurrences
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate scores
        total_count = positive_count + negative_count
        if total_count > 0:
            positive_score = positive_count / total_count
            negative_score = negative_count / total_count
            neutral_score = 0.0
            
            # Compound score: -1 (very negative) to 1 (very positive)
            compound_score = (positive_score - negative_score) * 2
        else:
            positive_score = 0.0
            negative_score = 0.0
            neutral_score = 1.0
            compound_score = 0.0
        
        return {
            "compound": compound_score,
            "positive": positive_score,
            "negative": negative_score,
            "neutral": neutral_score
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of extracted entities with metadata
        """
        # Ensure NLP pipeline is initialized
        if not self.nlp:
            self._initialize_nlp_pipeline()
            if not self.nlp:
                return []
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract entities
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
            
            return entities
        
        except Exception as e:
            logger.warning(f"Error in entity extraction: {str(e)}")
            return []
    
    def categorize_content(self, text: str) -> Dict[str, float]:
        """
        Categorize content into predefined categories
        
        Args:
            text: Text to categorize
            
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
            
        # Filter news by recency
        filtered_news = self.filter_by_recency(
            news_items, 
            timestamp_field="timestamp",
            lookback_hours=self.parameters["lookback_hours"]
        )
                
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
            categories = self.categorize_content(text)
            
            # Assess potential impact on currency pairs
            pair_impacts = self.assess_market_impact(text, categories, entities)
            
            # Calculate overall impact score
            max_impact = max([impact["impact_score"] for impact in pair_impacts.values()]) if pair_impacts else 0
            
            # Create sentiment result
            sentiment = self.analyze_sentiment(text)
            
            news_result = SentimentResult(
                id=news.get("id", ""),
                title=news.get("title", ""),
                source=news.get("source", ""),
                timestamp=news.get("timestamp", ""),
                content_snippet=news.get("content", "")[:200] + "..." if len(news.get("content", "")) > 200 else news.get("content", ""),
                compound_score=sentiment["compound"],
                positive_score=sentiment["positive"],
                negative_score=sentiment["negative"],
                neutral_score=sentiment["neutral"],
                categories=categories,
                entities=entities,
                market_impacts=pair_impacts,
                overall_impact_score=max_impact
            )
            
            results.append(news_result.to_dict())
            
        # Sort results by impact score
        results.sort(key=lambda x: x["overall_impact_score"], reverse=True)
        
        # Identify high-impact news
        impact_threshold = self.parameters["impact_threshold"]
        high_impact_news = [news for news in results if news["overall_impact_score"] >= impact_threshold]
        
        # Summary by currency pair
        pair_summary = {}
        for news in results:
            for pair, impact in news["market_impacts"].items():
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