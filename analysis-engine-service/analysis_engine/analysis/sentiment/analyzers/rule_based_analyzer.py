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


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class RuleBasedSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Sentiment analyzer using rule-based techniques.
    
    This analyzer uses predefined rules and patterns to analyze sentiment
    and assess market impact, focusing on explicit rules rather than
    statistical or machine learning approaches.
    """

    def __init__(self, parameters: Dict[str, Any]=None):
        """
        Initialize the rule-based sentiment analyzer
        
        Args:
            parameters: Configuration parameters for the analyzer
        """
        default_params = {'impact_threshold': 0.6, 'lookback_hours': 24,
            'sentiment_rules': [[
            '\\bstrong (growth|increase|gain|rally)\\b', 0.8, 1.0], [
            '\\bsurge[ds]?\\b', 0.7, 1.0], ['\\bexceed[s]? expectations\\b',
            0.8, 1.0], ['\\bbeat[s]? forecast\\b', 0.7, 1.0], [
            '\\brecord high\\b', 0.8, 1.0], [
            '\\bvery (bullish|positive)\\b', 0.9, 1.0], [
            '\\bincrease[ds]?\\b', 0.5, 0.8], ['\\bgrowth\\b', 0.5, 0.8], [
            '\\bimprove[ds]?\\b', 0.5, 0.8], ['\\bpositive\\b', 0.5, 0.8],
            ['\\bbullish\\b', 0.6, 0.8], ['\\bgain[s]?\\b', 0.5, 0.8], [
            '\\bsharp (drop|decline|decrease|fall)\\b', -0.8, 1.0], [
            '\\bplunge[ds]?\\b', -0.7, 1.0], [
            '\\bmiss[es]? expectations\\b', -0.8, 1.0], [
            '\\bfail[s]? to meet\\b', -0.7, 1.0], ['\\brecord low\\b', -0.8,
            1.0], ['\\bvery (bearish|negative)\\b', -0.9, 1.0], [
            '\\bdecrease[ds]?\\b', -0.5, 0.8], ['\\bdecline[ds]?\\b', -0.5,
            0.8], ['\\bweaken[s]?\\b', -0.5, 0.8], ['\\bnegative\\b', -0.5,
            0.8], ['\\bbearish\\b', -0.6, 0.8], ['\\bloss[es]?\\b', -0.5, 
            0.8], ['\\bunchanged\\b', 0.0, 0.5], ['\\bstable\\b', 0.0, 0.5],
            ['\\bflat\\b', 0.0, 0.5], ['\\bmixed\\b', 0.0, 0.5], [
            '\\bnot\\b', -1.0, 0.5], ['\\bno\\b', -1.0, 0.5]],
            'category_rules': [['monetary_policy',
            '\\binterest rate[s]?\\b', 1.0], ['monetary_policy',
            '\\bcentral bank\\b', 1.0], ['monetary_policy',
            '\\bmonetary policy\\b', 1.0], ['monetary_policy',
            '\\brate (decision|hike|cut)\\b', 1.0], ['monetary_policy',
            '\\bfed\\b', 0.8], ['monetary_policy', '\\becb\\b', 0.8], [
            'monetary_policy', '\\bboe\\b', 0.8], ['monetary_policy',
            '\\bboj\\b', 0.8], ['economic_indicators', '\\bgdp\\b', 1.0], [
            'economic_indicators', '\\binflation\\b', 1.0], [
            'economic_indicators', '\\bcpi\\b', 1.0], [
            'economic_indicators', '\\bppi\\b', 1.0], [
            'economic_indicators', '\\b(un)?employment\\b', 1.0], [
            'economic_indicators', '\\bnonfarm payroll\\b', 1.0], [
            'economic_indicators', '\\bretail sales\\b', 1.0], [
            'geopolitical', '\\bwar\\b', 1.0], ['geopolitical',
            '\\bconflict\\b', 1.0], ['geopolitical', '\\bsanctions\\b', 1.0
            ], ['geopolitical', '\\btariff[s]?\\b', 1.0], ['geopolitical',
            '\\btrade war\\b', 1.0], ['geopolitical', '\\belection\\b', 1.0
            ], ['geopolitical', '\\bbrexit\\b', 1.0], ['company_news',
            '\\bearnings\\b', 1.0], ['company_news', '\\bprofit[s]?\\b', 
            1.0], ['company_news', '\\brevenue[s]?\\b', 1.0], [
            'company_news', '\\bforecast\\b', 1.0], ['company_news',
            '\\boutlook\\b', 1.0], ['company_news', '\\bmerger\\b', 1.0], [
            'company_news', '\\bacquisition\\b', 1.0], ['market_sentiment',
            '\\brisk\\b', 1.0], ['market_sentiment', '\\brally\\b', 1.0], [
            'market_sentiment', '\\bsell-off\\b', 1.0], ['market_sentiment',
            '\\bcorrection\\b', 1.0], ['market_sentiment',
            '\\b(bull|bear)(ish)?\\b', 1.0], ['market_sentiment',
            '\\bsentiment\\b', 1.0], ['market_sentiment',
            '\\bvolatility\\b', 1.0]], 'currency_rules': [['USD',
            '\\bdollar\\b', 1.0], ['USD', '\\busd\\b', 1.0], ['USD',
            '\\bfederal reserve\\b', 0.8], ['USD', '\\bfed\\b', 0.8], [
            'USD', '\\bpowell\\b', 0.8], ['USD', '\\bus economy\\b', 0.8],
            ['EUR', '\\beuro\\b', 1.0], ['EUR', '\\beur\\b', 1.0], ['EUR',
            '\\becb\\b', 0.8], ['EUR', '\\beuropean central bank\\b', 0.8],
            ['EUR', '\\blagarde\\b', 0.8], ['EUR', '\\beurozone\\b', 0.8],
            ['GBP', '\\bpound\\b', 1.0], ['GBP', '\\bgbp\\b', 1.0], ['GBP',
            '\\bbank of england\\b', 0.8], ['GBP', '\\bboe\\b', 0.8], [
            'GBP', '\\bbailey\\b', 0.8], ['GBP', '\\buk economy\\b', 0.8],
            ['JPY', '\\byen\\b', 1.0], ['JPY', '\\bjpy\\b', 1.0], ['JPY',
            '\\bbank of japan\\b', 0.8], ['JPY', '\\bboj\\b', 0.8], ['JPY',
            '\\bkuroda\\b', 0.8], ['JPY', '\\bjapanese economy\\b', 0.8], [
            'AUD', '\\baussie\\b', 1.0], ['AUD', '\\baud\\b', 1.0], ['AUD',
            '\\breserve bank of australia\\b', 0.8], ['AUD', '\\brba\\b', 
            0.8], ['AUD', '\\blowe\\b', 0.8], ['AUD',
            '\\baustralian economy\\b', 0.8], ['NZD', '\\bkiwi\\b', 1.0], [
            'NZD', '\\bnzd\\b', 1.0], ['NZD',
            '\\breserve bank of new zealand\\b', 0.8], ['NZD', '\\brbnz\\b',
            0.8], ['NZD', '\\borr\\b', 0.8], ['NZD',
            '\\bnew zealand economy\\b', 0.8], ['CAD', '\\bloonie\\b', 1.0],
            ['CAD', '\\bcad\\b', 1.0], ['CAD', '\\bbank of canada\\b', 0.8],
            ['CAD', '\\bboc\\b', 0.8], ['CAD', '\\bmacklem\\b', 0.8], [
            'CAD', '\\bcanadian economy\\b', 0.8], ['CHF', '\\bfranc\\b', 
            1.0], ['CHF', '\\bchf\\b', 1.0], ['CHF',
            '\\bswiss national bank\\b', 0.8], ['CHF', '\\bsnb\\b', 0.8], [
            'CHF', '\\bjordan\\b', 0.8], ['CHF', '\\bswiss economy\\b', 0.8
            ]], 'impact_rules': [['\\bsignificant impact\\b', 0.8, 1.0], [
            '\\bmajor (impact|effect)\\b', 0.8, 1.0], [
            '\\bsubstantial (impact|effect)\\b', 0.7, 1.0], [
            '\\bmoderate (impact|effect)\\b', 0.5, 1.0], [
            '\\bminor (impact|effect)\\b', 0.3, 1.0], [
            '\\blimited (impact|effect)\\b', 0.2, 1.0], [
            '\\bno (impact|effect)\\b', 0.0, 1.0]]}
        merged_params = {**default_params, **parameters or {}}
        super().__init__('rule_based_sentiment_analyzer', merged_params)

    @with_analysis_resilience('analyze_sentiment')
    def analyze_sentiment(self, text: str) ->Dict[str, Any]:
        """
        Analyze sentiment using rule-based techniques
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        text_lower = text.lower()
        sentiment_rules = self.parameters['sentiment_rules']
        positive_score = 0.0
        negative_score = 0.0
        total_weight = 0.0
        for pattern, score, weight in sentiment_rules:
            matches = re.findall(pattern, text_lower)
            if matches:
                for match in matches:
                    match_pos = text_lower.find(match)
                    context = text_lower[max(0, match_pos - 20):min(len(
                        text_lower), match_pos + len(match) + 20)]
                    negated = any(re.search(
                        '\\b(not|no|never|neither|nor|without)\\b', context))
                    if negated:
                        applied_score = -score
                    else:
                        applied_score = score
                    if applied_score > 0:
                        positive_score += applied_score * weight
                    else:
                        negative_score += abs(applied_score) * weight
                    total_weight += weight
        if total_weight > 0:
            positive_score /= total_weight
            negative_score /= total_weight
        neutral_score = 1.0 - (positive_score + negative_score)
        neutral_score = max(0.0, neutral_score)
        compound_score = positive_score - negative_score
        return {'compound': compound_score, 'positive': positive_score,
            'negative': negative_score, 'neutral': neutral_score}

    def extract_entities(self, text: str) ->List[Dict[str, Any]]:
        """
        Extract entities using rule-based techniques
        
        Args:
            text: Text to analyze
            
        Returns:
            List of extracted entities with metadata
        """
        text_lower = text.lower()
        entities = []
        currency_rules = self.parameters['currency_rules']
        for currency, pattern, weight in currency_rules:
            for match in re.finditer(pattern, text_lower):
                entities.append({'text': match.group(), 'label': 'CURRENCY',
                    'entity_type': currency, 'start': match.start(), 'end':
                    match.end(), 'confidence': weight})
        for match in re.finditer('\\b\\d+(?:\\.\\d+)?%?\\b', text_lower):
            entities.append({'text': match.group(), 'label': 'NUMERIC',
                'start': match.start(), 'end': match.end(), 'confidence': 1.0})
        date_patterns = ['\\b\\d{1,2}[-/]\\d{1,2}[-/]\\d{2,4}\\b',
            '\\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \\d{1,2}(?:st|nd|rd|th)?,? \\d{2,4}\\b'
            , '\\b(?:yesterday|today|tomorrow)\\b',
            '\\blast (?:week|month|year)\\b', '\\bnext (?:week|month|year)\\b']
        for pattern in date_patterns:
            for match in re.finditer(pattern, text_lower):
                entities.append({'text': match.group(), 'label': 'DATE',
                    'start': match.start(), 'end': match.end(),
                    'confidence': 1.0})
        return entities

    def categorize_content(self, text: str) ->Dict[str, float]:
        """
        Categorize content using rule-based techniques
        
        Args:
            text: Text to categorize
            
        Returns:
            Dictionary mapping categories to confidence scores
        """
        text_lower = text.lower()
        category_rules = self.parameters['category_rules']
        categories = {}
        category_weights = {}
        for category, pattern, weight in category_rules:
            matches = re.findall(pattern, text_lower)
            if matches:
                if category not in categories:
                    categories[category] = 0.0
                    category_weights[category] = 0.0
                categories[category] += len(matches) * weight
                category_weights[category] += weight
        for category in categories:
            if category_weights[category] > 0:
                categories[category] = min(categories[category] /
                    category_weights[category], 1.0)
        return categories

    def assess_market_impact(self, text: str, categories: Dict[str, float],
        entities: List[Dict[str, Any]]) ->Dict[str, Dict[str, float]]:
        """
        Assess potential market impact using rule-based techniques
        
        Args:
            text: Text to analyze
            categories: Content categories with scores
            entities: Extracted entities
            
        Returns:
            Dictionary mapping market instruments to impact scores and directions
        """
        mentioned_currencies = set()
        for entity in entities:
            if entity.get('label') == 'CURRENCY':
                mentioned_currencies.add(entity.get('entity_type'))
        if not mentioned_currencies:
            mentioned_currencies = {'USD', 'EUR'}
        sentiment = self.analyze_sentiment(text)
        impact_score = 0.0
        impact_weight = 0.0
        impact_rules = self.parameters['impact_rules']
        text_lower = text.lower()
        for pattern, score, weight in impact_rules:
            matches = re.findall(pattern, text_lower)
            if matches:
                impact_score += len(matches) * score * weight
                impact_weight += len(matches) * weight
        if impact_weight == 0:
            impact_score = max(categories.values()) if categories else 0.5
        else:
            impact_score /= impact_weight
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
                magnitude = impact_score * abs(sentiment['compound'])
                pair_name = f'{base}/{quote}'
                pair_impacts[pair_name] = {'impact_score': magnitude,
                    'direction': direction, 'sentiment': sentiment[
                    'compound'], 'categories': {k: v for k, v in categories
                    .items() if v > 0.2}}
        return pair_impacts

    def analyze(self, data: Dict[str, Any]) ->AnalysisResult:
        """
        Analyze news data using rule-based techniques
        
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
