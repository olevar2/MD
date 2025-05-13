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


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class NewsAnalyzer(BaseNLPAnalyzer):
    """
    Analyzer for financial news content with market impact assessment.
    
    This analyzer processes financial news articles, extracts relevant entities,
    analyzes sentiment, and assesses potential market impact on forex pairs.
    """

    def __init__(self, parameters: Dict[str, Any]=None):
        """
        Initialize the news analyzer
        
        Args:
            parameters: Configuration parameters for the analyzer
        """
        default_params = {'spacy_model': 'en_core_web_sm',
            'impact_threshold': 0.6, 'lookback_hours': 24,
            'currency_sensitivity_map': {'USD': ['federal_reserve',
            'interest_rates', 'inflation', 'employment', 'gdp', 'fomc',
            'trade_balance'], 'EUR': ['ecb', 'eurozone', 'euro',
            'european_union', 'draghi', 'lagarde', 'france', 'germany'],
            'GBP': ['boe', 'brexit', 'uk', 'bank_of_england', 'trade_deal'],
            'JPY': ['boj', 'japan', 'kuroda', 'abenomics', 'safe_haven'],
            'AUD': ['rba', 'australia', 'commodities', 'china_trade'],
            'NZD': ['rbnz', 'new_zealand', 'dairy'], 'CAD': ['boc',
            'canada', 'oil', 'natural_resources'], 'CHF': ['snb', 'swiss',
            'safe_haven', 'swiss_national_bank']}}
        merged_params = {**default_params, **parameters or {}}
        super().__init__('news_analyzer', merged_params)
        self._initialize_news_classifiers()

    def _initialize_news_classifiers(self):
        """Initialize the news category classifiers"""
        self.category_keywords = {'monetary_policy': ['interest rate',
            'central bank', 'monetary policy', 'rate decision', 'rate hike',
            'rate cut', 'policy meeting', 'fomc', 'committee',
            'basis points', 'hawkish', 'dovish'], 'economic_indicators': [
            'gdp', 'inflation', 'cpi', 'ppi', 'employment', 'unemployment',
            'nonfarm', 'payroll', 'jobs', 'retail sales',
            'industrial production', 'manufacturing', 'pmi', 'ism',
            'trade balance', 'deficit', 'surplus'], 'geopolitical': ['war',
            'conflict', 'sanctions', 'tariff', 'trade war', 'election',
            'vote', 'referendum', 'brexit', 'political', 'government',
            'shutdown'], 'company_news': ['earnings', 'profit', 'revenue',
            'guidance', 'forecast', 'outlook', 'acquisition', 'merger',
            'takeover', 'ipo', 'stock', 'share'], 'market_sentiment': [
            'risk', 'rally', 'sell-off', 'correction', 'bull', 'bear',
            'sentiment', 'mood', 'fear', 'greed', 'panic', 'optimism',
            'pessimism', 'volatility']}

    def categorize_news(self, text: str) ->Dict[str, float]:
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
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            score = min(matches / (len(keywords) * 0.3), 1.0)
            scores[category] = score
        return scores

    def assess_pair_impact(self, text: str, categories: Dict[str, float],
        entities: List[Dict[str, Any]]) ->Dict[str, Dict[str, float]]:
        """
        Assess potential impact on currency pairs
        
        Args:
            text: News text
            categories: News categories with scores
            entities: Extracted entities
            
        Returns:
            Dictionary mapping currency pairs to impact scores and directions
        """
        mentioned_currencies = set()
        currency_sensitivity_map = self.parameters['currency_sensitivity_map']
        text_lower = text.lower()
        for currency, topics in currency_sensitivity_map.items():
            if currency.lower() in text_lower or any(topic.lower() in
                text_lower for topic in topics):
                mentioned_currencies.add(currency)
        for entity in entities:
            if entity['label'] == 'MONEY' or entity['label'] == 'ORG':
                for currency in currency_sensitivity_map.keys():
                    if currency.lower() in entity['text'].lower():
                        mentioned_currencies.add(currency)
        if not mentioned_currencies:
            mentioned_currencies = {'USD', 'EUR'}
        sentiment = self.analyze_sentiment(text)
        pair_impacts = {}
        major_pairs = [('EUR', 'USD'), ('USD', 'JPY'), ('GBP', 'USD'), (
            'USD', 'CHF'), ('USD', 'CAD'), ('AUD', 'USD'), ('NZD', 'USD')]
        for base, quote in major_pairs:
            if base in mentioned_currencies or quote in mentioned_currencies:
                direction = 0.0
                if (base in mentioned_currencies and quote in
                    mentioned_currencies):
                    direction = 0.2 * sentiment['compound']
                elif base in mentioned_currencies:
                    direction = sentiment['compound']
                elif quote in mentioned_currencies:
                    direction = -sentiment['compound']
                magnitude = max(categories.values()) * abs(sentiment[
                    'compound'])
                pair_name = f'{base}/{quote}'
                pair_impacts[pair_name] = {'impact_score': magnitude,
                    'direction': direction, 'sentiment': sentiment[
                    'compound'], 'categories': {k: v for k, v in categories
                    .items() if v > 0.2}}
        return pair_impacts

    @with_exception_handling
    def analyze(self, data: Dict[str, Any]) ->AnalysisResult:
        """
        Analyze news data
        
        Args:
            data: Dictionary containing news articles with text and metadata
            
        Returns:
            AnalysisResult containing analysis results
        """
        if not data or 'news_items' not in data:
            return AnalysisResult(analyzer_name=self.name, result_data={
                'error': 'No news data provided'}, is_valid=False)
        news_items = data['news_items']
        if not isinstance(news_items, list) or not news_items:
            return AnalysisResult(analyzer_name=self.name, result_data={
                'error': 'Invalid news data format or empty news list'},
                is_valid=False)
        filtered_news = []
        lookback_hours = self.parameters['lookback_hours']
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        for news in news_items:
            if 'timestamp' in news:
                try:
                    news_time = datetime.fromisoformat(news['timestamp'].
                        replace('Z', '+00:00'))
                    if news_time >= cutoff_time:
                        filtered_news.append(news)
                except (ValueError, TypeError):
                    filtered_news.append(news)
            else:
                filtered_news.append(news)
        if not filtered_news:
            return AnalysisResult(analyzer_name=self.name, result_data={
                'message': 'No recent news found'}, is_valid=True)
        results = []
        for news in filtered_news:
            text = news.get('title', '') + ' ' + news.get('content', '')
            if not text.strip():
                continue
            entities = self.extract_entities(text)
            categories = self.categorize_news(text)
            pair_impacts = self.assess_pair_impact(text, categories, entities)
            max_impact = max([impact['impact_score'] for impact in
                pair_impacts.values()]) if pair_impacts else 0
            news_result = {'id': news.get('id', ''), 'title': news.get(
                'title', ''), 'source': news.get('source', ''), 'timestamp':
                news.get('timestamp', ''), 'sentiment': self.
                analyze_sentiment(text), 'categories': categories,
                'entities': entities, 'pair_impacts': pair_impacts,
                'overall_impact_score': max_impact}
            results.append(news_result)
        results.sort(key=lambda x: x['overall_impact_score'], reverse=True)
        impact_threshold = self.parameters['impact_threshold']
        high_impact_news = [news for news in results if news[
            'overall_impact_score'] >= impact_threshold]
        pair_summary = {}
        for news in results:
            for pair, impact in news['pair_impacts'].items():
                if pair not in pair_summary:
                    pair_summary[pair] = {'count': 0, 'avg_impact': 0,
                        'avg_direction': 0, 'high_impact_count': 0}
                pair_summary[pair]['count'] += 1
                pair_summary[pair]['avg_impact'] += impact['impact_score']
                pair_summary[pair]['avg_direction'] += impact['direction']
                if news['overall_impact_score'] >= impact_threshold:
                    pair_summary[pair]['high_impact_count'] += 1
        for pair in pair_summary:
            count = pair_summary[pair]['count']
            if count > 0:
                pair_summary[pair]['avg_impact'] /= count
                pair_summary[pair]['avg_direction'] /= count
        return AnalysisResult(analyzer_name=self.name, result_data={
            'news_count': len(results), 'high_impact_count': len(
            high_impact_news), 'news_items': results, 'high_impact_news':
            high_impact_news, 'pair_summary': pair_summary}, is_valid=True)
