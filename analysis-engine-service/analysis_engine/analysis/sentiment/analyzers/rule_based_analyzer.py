"""
Rule-based Sentiment Analyzer

This module provides sentiment analysis using rule-based techniques.
"""

from typing import Dict, List, Any, Union
import logging
import re
from datetime import datetime, timedelta

from analysis_engine.analysis.sentiment.base_sentiment_analyzer import BaseSentimentAnalyzer
from analysis_engine.analysis.sentiment.models.sentiment_result import SentimentResult
from analysis_engine.models.analysis_result import AnalysisResult

logger = logging.getLogger(__name__)


class RuleBasedSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Sentiment analyzer using rule-based techniques.
    
    This analyzer uses predefined rules and patterns to analyze sentiment
    and assess market impact, focusing on explicit rules rather than
    statistical or machine learning approaches.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the rule-based sentiment analyzer
        
        Args:
            parameters: Configuration parameters for the analyzer
        """
        default_params = {
            "impact_threshold": 0.6,  # Threshold for considering news as impactful
            "lookback_hours": 24,  # Hours of news to analyze
            "sentiment_rules": [
                # Format: [pattern, sentiment_score, weight]
                # Strongly positive patterns
                [r'\bstrong (growth|increase|gain|rally)\b', 0.8, 1.0],
                [r'\bsurge[ds]?\b', 0.7, 1.0],
                [r'\bexceed[s]? expectations\b', 0.8, 1.0],
                [r'\bbeat[s]? forecast\b', 0.7, 1.0],
                [r'\brecord high\b', 0.8, 1.0],
                [r'\bvery (bullish|positive)\b', 0.9, 1.0],
                
                # Moderately positive patterns
                [r'\bincrease[ds]?\b', 0.5, 0.8],
                [r'\bgrowth\b', 0.5, 0.8],
                [r'\bimprove[ds]?\b', 0.5, 0.8],
                [r'\bpositive\b', 0.5, 0.8],
                [r'\bbullish\b', 0.6, 0.8],
                [r'\bgain[s]?\b', 0.5, 0.8],
                
                # Strongly negative patterns
                [r'\bsharp (drop|decline|decrease|fall)\b', -0.8, 1.0],
                [r'\bplunge[ds]?\b', -0.7, 1.0],
                [r'\bmiss[es]? expectations\b', -0.8, 1.0],
                [r'\bfail[s]? to meet\b', -0.7, 1.0],
                [r'\brecord low\b', -0.8, 1.0],
                [r'\bvery (bearish|negative)\b', -0.9, 1.0],
                
                # Moderately negative patterns
                [r'\bdecrease[ds]?\b', -0.5, 0.8],
                [r'\bdecline[ds]?\b', -0.5, 0.8],
                [r'\bweaken[s]?\b', -0.5, 0.8],
                [r'\bnegative\b', -0.5, 0.8],
                [r'\bbearish\b', -0.6, 0.8],
                [r'\bloss[es]?\b', -0.5, 0.8],
                
                # Neutral patterns
                [r'\bunchanged\b', 0.0, 0.5],
                [r'\bstable\b', 0.0, 0.5],
                [r'\bflat\b', 0.0, 0.5],
                [r'\bmixed\b', 0.0, 0.5],
                
                # Negation patterns (these modify the sentiment of nearby patterns)
                [r'\bnot\b', -1.0, 0.5],  # Reverses sentiment
                [r'\bno\b', -1.0, 0.5],   # Reverses sentiment
            ],
            "category_rules": [
                # Format: [category, pattern, weight]
                ["monetary_policy", r'\binterest rate[s]?\b', 1.0],
                ["monetary_policy", r'\bcentral bank\b', 1.0],
                ["monetary_policy", r'\bmonetary policy\b', 1.0],
                ["monetary_policy", r'\brate (decision|hike|cut)\b', 1.0],
                ["monetary_policy", r'\bfed\b', 0.8],
                ["monetary_policy", r'\becb\b', 0.8],
                ["monetary_policy", r'\bboe\b', 0.8],
                ["monetary_policy", r'\bboj\b', 0.8],
                
                ["economic_indicators", r'\bgdp\b', 1.0],
                ["economic_indicators", r'\binflation\b', 1.0],
                ["economic_indicators", r'\bcpi\b', 1.0],
                ["economic_indicators", r'\bppi\b', 1.0],
                ["economic_indicators", r'\b(un)?employment\b', 1.0],
                ["economic_indicators", r'\bnonfarm payroll\b', 1.0],
                ["economic_indicators", r'\bretail sales\b', 1.0],
                
                ["geopolitical", r'\bwar\b', 1.0],
                ["geopolitical", r'\bconflict\b', 1.0],
                ["geopolitical", r'\bsanctions\b', 1.0],
                ["geopolitical", r'\btariff[s]?\b', 1.0],
                ["geopolitical", r'\btrade war\b', 1.0],
                ["geopolitical", r'\belection\b', 1.0],
                ["geopolitical", r'\bbrexit\b', 1.0],
                
                ["company_news", r'\bearnings\b', 1.0],
                ["company_news", r'\bprofit[s]?\b', 1.0],
                ["company_news", r'\brevenue[s]?\b', 1.0],
                ["company_news", r'\bforecast\b', 1.0],
                ["company_news", r'\boutlook\b', 1.0],
                ["company_news", r'\bmerger\b', 1.0],
                ["company_news", r'\bacquisition\b', 1.0],
                
                ["market_sentiment", r'\brisk\b', 1.0],
                ["market_sentiment", r'\brally\b', 1.0],
                ["market_sentiment", r'\bsell-off\b', 1.0],
                ["market_sentiment", r'\bcorrection\b', 1.0],
                ["market_sentiment", r'\b(bull|bear)(ish)?\b', 1.0],
                ["market_sentiment", r'\bsentiment\b', 1.0],
                ["market_sentiment", r'\bvolatility\b', 1.0],
            ],
            "currency_rules": [
                # Format: [currency, pattern, weight]
                ["USD", r'\bdollar\b', 1.0],
                ["USD", r'\busd\b', 1.0],
                ["USD", r'\bfederal reserve\b', 0.8],
                ["USD", r'\bfed\b', 0.8],
                ["USD", r'\bpowell\b', 0.8],
                ["USD", r'\bus economy\b', 0.8],
                
                ["EUR", r'\beuro\b', 1.0],
                ["EUR", r'\beur\b', 1.0],
                ["EUR", r'\becb\b', 0.8],
                ["EUR", r'\beuropean central bank\b', 0.8],
                ["EUR", r'\blagarde\b', 0.8],
                ["EUR", r'\beurozone\b', 0.8],
                
                ["GBP", r'\bpound\b', 1.0],
                ["GBP", r'\bgbp\b', 1.0],
                ["GBP", r'\bbank of england\b', 0.8],
                ["GBP", r'\bboe\b', 0.8],
                ["GBP", r'\bbailey\b', 0.8],
                ["GBP", r'\buk economy\b', 0.8],
                
                ["JPY", r'\byen\b', 1.0],
                ["JPY", r'\bjpy\b', 1.0],
                ["JPY", r'\bbank of japan\b', 0.8],
                ["JPY", r'\bboj\b', 0.8],
                ["JPY", r'\bkuroda\b', 0.8],
                ["JPY", r'\bjapanese economy\b', 0.8],
                
                ["AUD", r'\baussie\b', 1.0],
                ["AUD", r'\baud\b', 1.0],
                ["AUD", r'\breserve bank of australia\b', 0.8],
                ["AUD", r'\brba\b', 0.8],
                ["AUD", r'\blowe\b', 0.8],
                ["AUD", r'\baustralian economy\b', 0.8],
                
                ["NZD", r'\bkiwi\b', 1.0],
                ["NZD", r'\bnzd\b', 1.0],
                ["NZD", r'\breserve bank of new zealand\b', 0.8],
                ["NZD", r'\brbnz\b', 0.8],
                ["NZD", r'\borr\b', 0.8],
                ["NZD", r'\bnew zealand economy\b', 0.8],
                
                ["CAD", r'\bloonie\b', 1.0],
                ["CAD", r'\bcad\b', 1.0],
                ["CAD", r'\bbank of canada\b', 0.8],
                ["CAD", r'\bboc\b', 0.8],
                ["CAD", r'\bmacklem\b', 0.8],
                ["CAD", r'\bcanadian economy\b', 0.8],
                
                ["CHF", r'\bfranc\b', 1.0],
                ["CHF", r'\bchf\b', 1.0],
                ["CHF", r'\bswiss national bank\b', 0.8],
                ["CHF", r'\bsnb\b', 0.8],
                ["CHF", r'\bjordan\b', 0.8],
                ["CHF", r'\bswiss economy\b', 0.8],
            ],
            "impact_rules": [
                # Format: [pattern, impact_score, weight]
                [r'\bsignificant impact\b', 0.8, 1.0],
                [r'\bmajor (impact|effect)\b', 0.8, 1.0],
                [r'\bsubstantial (impact|effect)\b', 0.7, 1.0],
                [r'\bmoderate (impact|effect)\b', 0.5, 1.0],
                [r'\bminor (impact|effect)\b', 0.3, 1.0],
                [r'\blimited (impact|effect)\b', 0.2, 1.0],
                [r'\bno (impact|effect)\b', 0.0, 1.0],
            ]
        }
        
        # Merge default parameters with provided parameters
        merged_params = {**default_params, **(parameters or {})}
        super().__init__("rule_based_sentiment_analyzer", merged_params)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using rule-based techniques
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        text_lower = text.lower()
        sentiment_rules = self.parameters["sentiment_rules"]
        
        # Initialize scores
        positive_score = 0.0
        negative_score = 0.0
        total_weight = 0.0
        
        # Apply sentiment rules
        for pattern, score, weight in sentiment_rules:
            matches = re.findall(pattern, text_lower)
            if matches:
                # Check for nearby negation
                for match in matches:
                    match_pos = text_lower.find(match)
                    context = text_lower[max(0, match_pos - 20):min(len(text_lower), match_pos + len(match) + 20)]
                    
                    # Check for negation in context
                    negated = any(re.search(r'\b(not|no|never|neither|nor|without)\b', context))
                    
                    # Apply score with negation if needed
                    if negated:
                        applied_score = -score
                    else:
                        applied_score = score
                    
                    # Update scores
                    if applied_score > 0:
                        positive_score += applied_score * weight
                    else:
                        negative_score += abs(applied_score) * weight
                    
                    total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            positive_score /= total_weight
            negative_score /= total_weight
        
        # Calculate neutral score
        neutral_score = 1.0 - (positive_score + negative_score)
        neutral_score = max(0.0, neutral_score)  # Ensure non-negative
        
        # Calculate compound score (-1 to 1)
        compound_score = positive_score - negative_score
        
        return {
            "compound": compound_score,
            "positive": positive_score,
            "negative": negative_score,
            "neutral": neutral_score
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using rule-based techniques
        
        Args:
            text: Text to analyze
            
        Returns:
            List of extracted entities with metadata
        """
        text_lower = text.lower()
        entities = []
        
        # Extract currency entities
        currency_rules = self.parameters["currency_rules"]
        
        for currency, pattern, weight in currency_rules:
            for match in re.finditer(pattern, text_lower):
                entities.append({
                    "text": match.group(),
                    "label": "CURRENCY",
                    "entity_type": currency,
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": weight
                })
        
        # Extract numeric entities (simplified)
        for match in re.finditer(r'\b\d+(?:\.\d+)?%?\b', text_lower):
            entities.append({
                "text": match.group(),
                "label": "NUMERIC",
                "start": match.start(),
                "end": match.end(),
                "confidence": 1.0
            })
        
        # Extract date entities (simplified)
        date_patterns = [
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # DD/MM/YYYY
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?,? \d{2,4}\b',  # Month DD, YYYY
            r'\b(?:yesterday|today|tomorrow)\b',
            r'\blast (?:week|month|year)\b',
            r'\bnext (?:week|month|year)\b'
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text_lower):
                entities.append({
                    "text": match.group(),
                    "label": "DATE",
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 1.0
                })
        
        return entities
    
    def categorize_content(self, text: str) -> Dict[str, float]:
        """
        Categorize content using rule-based techniques
        
        Args:
            text: Text to categorize
            
        Returns:
            Dictionary mapping categories to confidence scores
        """
        text_lower = text.lower()
        category_rules = self.parameters["category_rules"]
        
        # Initialize category scores
        categories = {}
        category_weights = {}
        
        # Apply category rules
        for category, pattern, weight in category_rules:
            matches = re.findall(pattern, text_lower)
            if matches:
                # Initialize category if not already present
                if category not in categories:
                    categories[category] = 0.0
                    category_weights[category] = 0.0
                
                # Update category score
                categories[category] += len(matches) * weight
                category_weights[category] += weight
        
        # Normalize category scores
        for category in categories:
            if category_weights[category] > 0:
                categories[category] = min(categories[category] / category_weights[category], 1.0)
        
        return categories
    
    def assess_market_impact(self, text: str, categories: Dict[str, float], 
                           entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Assess potential market impact using rule-based techniques
        
        Args:
            text: Text to analyze
            categories: Content categories with scores
            entities: Extracted entities
            
        Returns:
            Dictionary mapping market instruments to impact scores and directions
        """
        # Extract mentioned currencies
        mentioned_currencies = set()
        
        for entity in entities:
            if entity.get("label") == "CURRENCY":
                mentioned_currencies.add(entity.get("entity_type"))
        
        # If no currencies detected, use major currencies as default
        if not mentioned_currencies:
            mentioned_currencies = {"USD", "EUR"}
        
        # Calculate sentiment
        sentiment = self.analyze_sentiment(text)
        
        # Calculate impact score
        impact_score = 0.0
        impact_weight = 0.0
        
        # Apply impact rules
        impact_rules = self.parameters["impact_rules"]
        text_lower = text.lower()
        
        for pattern, score, weight in impact_rules:
            matches = re.findall(pattern, text_lower)
            if matches:
                impact_score += len(matches) * score * weight
                impact_weight += len(matches) * weight
        
        # If no explicit impact mentioned, use category scores
        if impact_weight == 0:
            impact_score = max(categories.values()) if categories else 0.5
        else:
            impact_score /= impact_weight
        
        # Generate potential impacts for currency pairs
        pair_impacts = {}
        
        # Major currency pairs
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
                    direction = 0.2 * sentiment["compound"]  # Dampened impact when both affected
                elif base in mentioned_currencies:
                    direction = sentiment["compound"]  # Base currency affected
                elif quote in mentioned_currencies:
                    direction = -sentiment["compound"]  # Quote currency affected (inverted)
                
                # Calculate impact magnitude
                magnitude = impact_score * abs(sentiment["compound"])
                
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
        Analyze news data using rule-based techniques
        
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