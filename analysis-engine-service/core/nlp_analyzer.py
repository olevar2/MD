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


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class NLPSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Sentiment analyzer using natural language processing techniques.
    
    This analyzer uses NLP to process text, extract entities, analyze sentiment,
    and assess potential market impact.
    """

    def __init__(self, parameters: Dict[str, Any]=None):
        """
        Initialize the NLP sentiment analyzer
        
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
        super().__init__('nlp_sentiment_analyzer', merged_params)
        self._initialize_nlp_pipeline()
        self._initialize_category_classifiers()

    @with_exception_handling
    def _initialize_nlp_pipeline(self):
        """Initialize the NLP pipeline"""
        try:
            import spacy
            self.nlp = spacy.load(self.parameters['spacy_model'])
            logger.info(f"Loaded spaCy model: {self.parameters['spacy_model']}"
                )
        except Exception as e:
            logger.warning(f'Failed to load spaCy model: {str(e)}')
            self.nlp = None

    def _initialize_category_classifiers(self):
        """Initialize the category classifiers"""
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

    @with_exception_handling
    def analyze_sentiment(self, text: str) ->Dict[str, Any]:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not self.nlp:
            self._initialize_nlp_pipeline()
            if not self.nlp:
                return self._basic_sentiment_analysis(text)
        try:
            doc = self.nlp(text)
            positive_words = set(['increase', 'growth', 'positive',
                'bullish', 'upward', 'gain', 'profit', 'strengthen',
                'improved', 'recovery', 'strong', 'robust', 'exceed'])
            negative_words = set(['decrease', 'decline', 'negative',
                'bearish', 'downward', 'loss', 'deficit', 'weaken',
                'deteriorate', 'recession', 'weak', 'below', 'fail'])
            positive_count = sum(1 for token in doc if token.text.lower() in
                positive_words)
            negative_count = sum(1 for token in doc if token.text.lower() in
                negative_words)
            total_count = len(doc)
            if total_count > 0:
                positive_score = positive_count / total_count
                negative_score = negative_count / total_count
                neutral_score = 1.0 - (positive_score + negative_score)
                compound_score = (positive_score - negative_score) * 2
            else:
                positive_score = 0.0
                negative_score = 0.0
                neutral_score = 1.0
                compound_score = 0.0
            return {'compound': compound_score, 'positive': positive_score,
                'negative': negative_score, 'neutral': neutral_score}
        except Exception as e:
            logger.warning(f'Error in sentiment analysis: {str(e)}')
            return self._basic_sentiment_analysis(text)

    def _basic_sentiment_analysis(self, text: str) ->Dict[str, Any]:
        """
        Basic sentiment analysis fallback
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        text_lower = text.lower()
        positive_words = ['increase', 'growth', 'positive', 'bullish',
            'upward', 'gain', 'profit', 'strengthen', 'improved',
            'recovery', 'strong', 'robust', 'exceed']
        negative_words = ['decrease', 'decline', 'negative', 'bearish',
            'downward', 'loss', 'deficit', 'weaken', 'deteriorate',
            'recession', 'weak', 'below', 'fail']
        positive_count = sum(1 for word in positive_words if word in text_lower
            )
        negative_count = sum(1 for word in negative_words if word in text_lower
            )
        total_count = positive_count + negative_count
        if total_count > 0:
            positive_score = positive_count / total_count
            negative_score = negative_count / total_count
            neutral_score = 0.0
            compound_score = (positive_score - negative_score) * 2
        else:
            positive_score = 0.0
            negative_score = 0.0
            neutral_score = 1.0
            compound_score = 0.0
        return {'compound': compound_score, 'positive': positive_score,
            'negative': negative_score, 'neutral': neutral_score}

    @with_exception_handling
    def extract_entities(self, text: str) ->List[Dict[str, Any]]:
        """
        Extract named entities from text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of extracted entities with metadata
        """
        if not self.nlp:
            self._initialize_nlp_pipeline()
            if not self.nlp:
                return []
        try:
            doc = self.nlp(text)
            entities = []
            for ent in doc.ents:
                entities.append({'text': ent.text, 'label': ent.label_,
                    'start': ent.start_char, 'end': ent.end_char})
            return entities
        except Exception as e:
            logger.warning(f'Error in entity extraction: {str(e)}')
            return []

    def categorize_content(self, text: str) ->Dict[str, float]:
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
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            score = min(matches / (len(keywords) * 0.3), 1.0)
            scores[category] = score
        return scores

    def assess_market_impact(self, text: str, categories: Dict[str, float],
        entities: List[Dict[str, Any]]) ->Dict[str, Dict[str, float]]:
        """
        Assess potential market impact
        
        Args:
            text: Text to analyze
            categories: Content categories with scores
            entities: Extracted entities
            
        Returns:
            Dictionary mapping market instruments to impact scores and directions
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
        filtered_news = self.filter_by_recency(news_items, timestamp_field=
            'timestamp', lookback_hours=self.parameters['lookback_hours'])
        if not filtered_news:
            return AnalysisResult(analyzer_name=self.name, result_data={
                'message': 'No recent news found'}, is_valid=True)
        results = []
        for news in filtered_news:
            text = news.get('title', '') + ' ' + news.get('content', '')
            if not text.strip():
                continue
            entities = self.extract_entities(text)
            categories = self.categorize_content(text)
            pair_impacts = self.assess_market_impact(text, categories, entities
                )
            max_impact = max([impact['impact_score'] for impact in
                pair_impacts.values()]) if pair_impacts else 0
            sentiment = self.analyze_sentiment(text)
            news_result = SentimentResult(id=news.get('id', ''), title=news
                .get('title', ''), source=news.get('source', ''), timestamp
                =news.get('timestamp', ''), content_snippet=news.get(
                'content', '')[:200] + '...' if len(news.get('content', '')
                ) > 200 else news.get('content', ''), compound_score=
                sentiment['compound'], positive_score=sentiment['positive'],
                negative_score=sentiment['negative'], neutral_score=
                sentiment['neutral'], categories=categories, entities=
                entities, market_impacts=pair_impacts, overall_impact_score
                =max_impact)
            results.append(news_result.to_dict())
        results.sort(key=lambda x: x['overall_impact_score'], reverse=True)
        impact_threshold = self.parameters['impact_threshold']
        high_impact_news = [news for news in results if news[
            'overall_impact_score'] >= impact_threshold]
        pair_summary = {}
        for news in results:
            for pair, impact in news['market_impacts'].items():
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
