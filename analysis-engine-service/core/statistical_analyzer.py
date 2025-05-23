"""
Statistical Sentiment Analyzer

This module provides sentiment analysis using statistical techniques.
"""
from typing import Dict, List, Any, Union
import logging
import re
from datetime import datetime, timedelta
import numpy as np
from analysis_engine.analysis.sentiment.base_sentiment_analyzer import BaseSentimentAnalyzer
from analysis_engine.analysis.sentiment.models.sentiment_result import SentimentResult
from analysis_engine.models.analysis_result import AnalysisResult
logger = logging.getLogger(__name__)


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class StatisticalSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Sentiment analyzer using statistical techniques.
    
    This analyzer uses statistical methods to analyze sentiment and assess
    market impact, focusing on quantitative patterns rather than linguistic features.
    """

    def __init__(self, parameters: Dict[str, Any]=None):
        """
        Initialize the statistical sentiment analyzer
        
        Args:
            parameters: Configuration parameters for the analyzer
        """
        default_params = {'impact_threshold': 0.6, 'lookback_hours': 24,
            'term_frequency_threshold': 0.01, 'sentiment_lexicon': {
            'positive': ['increase', 'growth', 'positive', 'bullish',
            'upward', 'gain', 'profit', 'strengthen', 'improved',
            'recovery', 'strong', 'robust', 'exceed', 'beat', 'surge',
            'rally', 'rise', 'higher', 'better', 'success'], 'negative': [
            'decrease', 'decline', 'negative', 'bearish', 'downward',
            'loss', 'deficit', 'weaken', 'deteriorate', 'recession', 'weak',
            'below', 'fail', 'miss', 'plunge', 'drop', 'fall', 'lower',
            'worse', 'risk']}, 'market_terms': {'USD': ['dollar', 'usd',
            'federal reserve', 'fed', 'powell', 'us economy',
            'united states'], 'EUR': ['euro', 'eur', 'ecb',
            'european central bank', 'lagarde', 'eurozone',
            'european union'], 'GBP': ['pound', 'gbp', 'bank of england',
            'boe', 'bailey', 'uk economy', 'britain', 'british'], 'JPY': [
            'yen', 'jpy', 'bank of japan', 'boj', 'kuroda',
            'japanese economy', 'japan'], 'AUD': ['aussie', 'aud',
            'reserve bank of australia', 'rba', 'lowe',
            'australian economy', 'australia'], 'NZD': ['kiwi', 'nzd',
            'reserve bank of new zealand', 'rbnz', 'orr',
            'new zealand economy'], 'CAD': ['loonie', 'cad',
            'bank of canada', 'boc', 'macklem', 'canadian economy',
            'canada'], 'CHF': ['franc', 'chf', 'swiss national bank', 'snb',
            'jordan', 'swiss economy', 'switzerland']}}
        merged_params = {**default_params, **parameters or {}}
        super().__init__('statistical_sentiment_analyzer', merged_params)
        self.term_frequencies = {}
        self.document_count = 0

    @with_analysis_resilience('analyze_sentiment')
    def analyze_sentiment(self, text: str) ->Dict[str, Any]:
        """
        Analyze sentiment of text using statistical methods
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        tokens = re.findall('\\b\\w+\\b', text.lower())
        sentiment_lexicon = self.parameters['sentiment_lexicon']
        positive_words = set(sentiment_lexicon['positive'])
        negative_words = set(sentiment_lexicon['negative'])
        positive_count = sum(1 for token in tokens if token in positive_words)
        negative_count = sum(1 for token in tokens if token in negative_words)
        total_count = len(tokens)
        positive_score = 0.0
        negative_score = 0.0
        if self.document_count > 0 and total_count > 0:
            for token in tokens:
                if token in positive_words:
                    tf = tokens.count(token) / total_count
                    idf = np.log(self.document_count / (1 + self.
                        term_frequencies.get(token, 1)))
                    weight = tf * idf
                    positive_score += weight
                elif token in negative_words:
                    tf = tokens.count(token) / total_count
                    idf = np.log(self.document_count / (1 + self.
                        term_frequencies.get(token, 1)))
                    weight = tf * idf
                    negative_score += weight
            max_score = max(positive_score, negative_score)
            if max_score > 0:
                positive_score /= max_score
                negative_score /= max_score
        elif total_count > 0:
            positive_score = positive_count / total_count
            negative_score = negative_count / total_count
        neutral_score = 1.0 - (positive_score + negative_score)
        neutral_score = max(0.0, neutral_score)
        compound_score = positive_score - negative_score
        return {'compound': compound_score, 'positive': positive_score,
            'negative': negative_score, 'neutral': neutral_score}

    @with_resilience('update_term_frequencies')
    def update_term_frequencies(self, text: str):
        """
        Update term frequency statistics
        
        Args:
            text: Text to analyze
        """
        tokens = re.findall('\\b\\w+\\b', text.lower())
        self.document_count += 1
        for token in set(tokens):
            self.term_frequencies[token] = self.term_frequencies.get(token, 0
                ) + 1

    def extract_entities(self, text: str) ->List[Dict[str, Any]]:
        """
        Extract entities using statistical methods
        
        Args:
            text: Text to analyze
            
        Returns:
            List of extracted entities with metadata
        """
        entities = []
        text_lower = text.lower()
        market_terms = self.parameters['market_terms']
        for entity_type, terms in market_terms.items():
            for term in terms:
                if term in text_lower:
                    for match in re.finditer('\\b' + re.escape(term) +
                        '\\b', text_lower):
                        entities.append({'text': term, 'label':
                            'MARKET_ENTITY', 'entity_type': entity_type,
                            'start': match.start(), 'end': match.end()})
        for match in re.finditer('\\b\\d+(?:\\.\\d+)?%?\\b', text_lower):
            entities.append({'text': match.group(), 'label': 'NUMERIC',
                'start': match.start(), 'end': match.end()})
        return entities

    def categorize_content(self, text: str) ->Dict[str, float]:
        """
        Categorize content using statistical methods
        
        Args:
            text: Text to categorize
            
        Returns:
            Dictionary mapping categories to confidence scores
        """
        category_keywords = {'monetary_policy': ['interest rate',
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
        text_lower = text.lower()
        scores = {}
        tokens = re.findall('\\b\\w+\\b', text_lower)
        total_tokens = len(tokens)
        for category, keywords in category_keywords.items():
            category_score = 0.0
            for keyword in keywords:
                count = text_lower.count(keyword)
                if count > 0 and total_tokens > 0:
                    tf = count / total_tokens
                    if self.document_count > 0:
                        first_word = keyword.split()[0]
                        idf = np.log(self.document_count / (1 + self.
                            term_frequencies.get(first_word, 1)))
                    else:
                        idf = 1.0
                    category_score += tf * idf
            scores[category] = min(category_score, 1.0)
        return scores

    def assess_market_impact(self, text: str, categories: Dict[str, float],
        entities: List[Dict[str, Any]]) ->Dict[str, Dict[str, float]]:
        """
        Assess potential market impact using statistical methods
        
        Args:
            text: Text to analyze
            categories: Content categories with scores
            entities: Extracted entities
            
        Returns:
            Dictionary mapping market instruments to impact scores and directions
        """
        mentioned_currencies = set()
        for entity in entities:
            if entity.get('label') == 'MARKET_ENTITY':
                mentioned_currencies.add(entity.get('entity_type'))
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
                category_factor = max(categories.values()
                    ) if categories else 0.5
                magnitude = category_factor * abs(sentiment['compound'])
                pair_name = f'{base}/{quote}'
                pair_impacts[pair_name] = {'impact_score': magnitude,
                    'direction': direction, 'sentiment': sentiment[
                    'compound'], 'categories': {k: v for k, v in categories
                    .items() if v > 0.2}}
        return pair_impacts

    def analyze(self, data: Dict[str, Any]) ->AnalysisResult:
        """
        Analyze news data using statistical methods
        
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
        for news in filtered_news:
            text = news.get('title', '') + ' ' + news.get('content', '')
            if text.strip():
                self.update_term_frequencies(text)
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
            high_impact_news, 'pair_summary': pair_summary,
            'document_count': self.document_count, 'term_count': len(self.
            term_frequencies)}, is_valid=True)
