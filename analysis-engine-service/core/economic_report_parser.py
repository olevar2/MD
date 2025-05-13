"""
Economic Report Parser Module

This module provides functionality for parsing and analyzing economic reports
such as NFP, GDP, CPI, and other market-moving economic data releases.
"""
from typing import Dict, List, Any, Union, Optional
import logging
import re
from datetime import datetime
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

class EconomicReportParser(BaseNLPAnalyzer):
    """
    Parser for economic reports and data releases.
    
    This analyzer extracts key metrics from economic reports, compares them
    to expectations, and assesses potential market impact on forex pairs.
    """

    def __init__(self, parameters: Dict[str, Any]=None):
        """
        Initialize the economic report parser
        
        Args:
            parameters: Configuration parameters for the analyzer
        """
        default_params = {'spacy_model': 'en_core_web_sm',
            'significant_deviation_threshold': 0.1, 'impact_mapping': {
            'NFP': {'high': ['USD']}, 'GDP': {'high': ['USD', 'EUR', 'GBP',
            'AUD', 'CAD', 'JPY']}, 'CPI': {'high': ['USD', 'EUR', 'GBP',
            'AUD', 'CAD'], 'medium': ['JPY', 'CHF']}, 'Retail Sales': {
            'high': ['USD', 'GBP', 'AUD'], 'medium': ['EUR', 'CAD']}, 'PMI':
            {'high': ['EUR'], 'medium': ['USD', 'GBP']},
            'Interest Rate Decision': {'high': ['USD', 'EUR', 'GBP', 'AUD',
            'CAD', 'JPY', 'CHF', 'NZD']}, 'Unemployment Rate': {'high': [
            'USD', 'EUR', 'GBP', 'AUD'], 'medium': ['CAD', 'JPY']},
            'Trade Balance': {'medium': ['USD', 'EUR', 'GBP', 'AUD', 'CAD',
            'JPY']}, 'Industrial Production': {'medium': ['USD', 'EUR',
            'GBP'], 'low': ['AUD', 'CAD']}}, 'currency_weight': {'USD': 1.0,
            'EUR': 0.9, 'JPY': 0.8, 'GBP': 0.7, 'AUD': 0.6, 'CAD': 0.6,
            'CHF': 0.6, 'NZD': 0.5}}
        merged_params = {**default_params, **parameters or {}}
        super().__init__('economic_report_parser', merged_params)

    @with_exception_handling
    def _extract_metrics(self, report_text: str) ->Dict[str, Any]:
        """
        Extract key metrics from report text using regex patterns
        
        Args:
            report_text: Text of the economic report
            
        Returns:
            Dictionary of extracted metrics
        """
        metrics = {}
        value_pattern = '(-?\\d+\\.?\\d*%?)'
        patterns = {'current': ['actual:?\\s*' + value_pattern, 
            'reported:?\\s*' + value_pattern, 'came in at:?\\s*' +
            value_pattern, 'released at:?\\s*' + value_pattern, 
            'current:?\\s*' + value_pattern, 'reading:?\\s*' +
            value_pattern], 'previous': ['previous:?\\s*' + value_pattern, 
            'prior:?\\s*' + value_pattern, 'last:?\\s*' + value_pattern, 
            'earlier:?\\s*' + value_pattern], 'forecast': ['forecast:?\\s*' +
            value_pattern, 'expected:?\\s*' + value_pattern, 
            'estimate:?\\s*' + value_pattern, 'consensus:?\\s*' +
            value_pattern, 'expectation:?\\s*' + value_pattern, 
            'projection:?\\s*' + value_pattern, 'anticipated:?\\s*' +
            value_pattern], 'revised': ['revised:?\\s*' + value_pattern, 
            'adjusted:?\\s*' + value_pattern, 'updated:?\\s*' + value_pattern]}
        for metric_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, report_text, re.IGNORECASE)
                if matches:
                    value_str = matches[0]
                    try:
                        if '%' in value_str:
                            value = float(value_str.replace('%', '')) / 100
                            metrics[metric_type] = {'value': value,
                                'display': value_str, 'is_percent': True}
                        else:
                            value = float(value_str)
                            metrics[metric_type] = {'value': value,
                                'display': value_str, 'is_percent': False}
                    except ValueError:
                        metrics[metric_type] = {'value': value_str,
                            'display': value_str, 'is_percent': '%' in
                            value_str}
                    break
        return metrics

    def _calculate_deviations(self, metrics: Dict[str, Any]) ->Dict[str, float
        ]:
        """
        Calculate deviations from forecast and previous values
        
        Args:
            metrics: Extracted metrics
            
        Returns:
            Dictionary of calculated deviations
        """
        deviations = {}
        current = metrics.get('current', {}).get('value')
        previous = metrics.get('previous', {}).get('value')
        forecast = metrics.get('forecast', {}).get('value')
        if isinstance(current, (int, float)):
            if isinstance(forecast, (int, float)) and forecast != 0:
                deviations['from_forecast'] = (current - forecast) / abs(
                    forecast)
                deviations['from_forecast_significant'] = abs(deviations[
                    'from_forecast']) >= self.parameters[
                    'significant_deviation_threshold']
            if isinstance(previous, (int, float)) and previous != 0:
                deviations['from_previous'] = (current - previous) / abs(
                    previous)
                deviations['from_previous_significant'] = abs(deviations[
                    'from_previous']) >= self.parameters[
                    'significant_deviation_threshold']
        return deviations

    def _determine_report_type(self, report_data: Dict[str, Any]) ->str:
        """
        Determine the type of economic report based on title and content
        
        Args:
            report_data: Report data including title and content
            
        Returns:
            Type of report (e.g., "NFP", "GDP", "CPI", etc.)
        """
        title = report_data.get('title', '').lower()
        content = report_data.get('content', '').lower()
        text = title + ' ' + content
        report_keywords = {'NFP': ['nonfarm payroll', 'nonfarm payrolls',
            'non-farm payroll', 'non farm payroll', 'jobs report'], 'GDP':
            ['gross domestic product', 'gdp', 'economic growth'], 'CPI': [
            'consumer price index', 'cpi', 'inflation', 'consumer prices'],
            'PPI': ['producer price index', 'ppi', 'producer prices'],
            'Retail Sales': ['retail sales', 'consumer spending'], 'PMI': [
            'purchasing managers index', 'pmi', 'manufacturing index',
            'services index'], 'Interest Rate Decision': ['interest rate',
            'rate decision', 'fomc', 'central bank', 'fed funds',
            'monetary policy'], 'Unemployment Rate': ['unemployment rate',
            'jobless', 'unemployment'], 'Trade Balance': ['trade balance',
            'trade deficit', 'trade surplus', 'imports', 'exports'],
            'Industrial Production': ['industrial production',
            'manufacturing output']}
        for report_type, keywords in report_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return report_type
        return 'Economic Report'

    def _determine_sentiment(self, deviations: Dict[str, float],
        report_type: str) ->Dict[str, Any]:
        """
        Determine sentiment based on deviations and report type
        
        Args:
            deviations: Calculated deviations
            report_type: Type of economic report
            
        Returns:
            Dictionary with sentiment information
        """
        sentiment = {'value': 0, 'label': 'neutral', 'confidence': 0.5}
        inverse_indicators = ['Unemployment Rate', 'CPI', 'PPI']
        invert_reading = report_type in inverse_indicators
        if 'from_forecast' in deviations:
            deviation = deviations['from_forecast']
            if invert_reading:
                deviation = -deviation
            sentiment['value'] = max(-1, min(1, deviation * 3))
            if abs(deviation) >= 0.2:
                sentiment['confidence'] = 0.9
            elif abs(deviation) >= 0.1:
                sentiment['confidence'] = 0.7
            else:
                sentiment['confidence'] = 0.5
            if sentiment['value'] > 0.3:
                sentiment['label'] = 'bullish'
            elif sentiment['value'] > 0.1:
                sentiment['label'] = 'slightly bullish'
            elif sentiment['value'] < -0.3:
                sentiment['label'] = 'bearish'
            elif sentiment['value'] < -0.1:
                sentiment['label'] = 'slightly bearish'
            else:
                sentiment['label'] = 'neutral'
        return sentiment

    def _assess_currency_impacts(self, report_type: str, sentiment: Dict[
        str, Any]) ->Dict[str, float]:
        """
        Assess impact on individual currencies
        
        Args:
            report_type: Type of economic report
            sentiment: Determined sentiment
            
        Returns:
            Dictionary mapping currencies to impact values
        """
        impact_mapping = self.parameters['impact_mapping']
        currency_impacts = {}
        report_impacts = impact_mapping.get(report_type, {})
        for impact_level, currencies in report_impacts.items():
            impact_multiplier = 1.0
            if impact_level == 'high':
                impact_multiplier = 1.0
            elif impact_level == 'medium':
                impact_multiplier = 0.7
            elif impact_level == 'low':
                impact_multiplier = 0.4
            for currency in currencies:
                currency_weight = self.parameters['currency_weight'].get(
                    currency, 0.5)
                impact = sentiment['value'
                    ] * impact_multiplier * currency_weight * sentiment[
                    'confidence']
                currency_impacts[currency] = impact
        return currency_impacts

    def _assess_pair_impacts(self, currency_impacts: Dict[str, float]) ->Dict[
        str, Dict[str, Any]]:
        """
        Assess impact on currency pairs based on individual currency impacts
        
        Args:
            currency_impacts: Impact on individual currencies
            
        Returns:
            Dictionary mapping currency pairs to impact information
        """
        pair_impacts = {}
        major_pairs = [('EUR', 'USD'), ('USD', 'JPY'), ('GBP', 'USD'), (
            'USD', 'CHF'), ('USD', 'CAD'), ('AUD', 'USD'), ('NZD', 'USD')]
        for base, quote in major_pairs:
            base_impact = currency_impacts.get(base, 0)
            quote_impact = currency_impacts.get(quote, 0)
            if base_impact == 0 and quote_impact == 0:
                continue
            if base == 'USD' and quote in ['JPY', 'CHF', 'CAD']:
                net_impact = base_impact - quote_impact
            else:
                net_impact = base_impact - quote_impact
            pair_name = f'{base}/{quote}'
            abs_impact = abs(net_impact)
            if abs_impact >= 0.5:
                strength = 'strong'
            elif abs_impact >= 0.2:
                strength = 'moderate'
            else:
                strength = 'mild'
            if net_impact > 0.1:
                direction = f'{pair_name} likely to rise'
            elif net_impact < -0.1:
                direction = f'{pair_name} likely to fall'
            else:
                direction = f'Limited impact on {pair_name}'
            pair_impacts[pair_name] = {'impact_value': net_impact,
                'impact_strength': strength, 'direction': direction,
                'base_currency_impact': base_impact,
                'quote_currency_impact': quote_impact}
        return pair_impacts

    def analyze(self, data: Dict[str, Any]) ->AnalysisResult:
        """
        Analyze economic report data
        
        Args:
            data: Dictionary containing economic report data
            
        Returns:
            AnalysisResult containing analysis results
        """
        if not data or 'report' not in data:
            return AnalysisResult(analyzer_name=self.name, result_data={
                'error': 'No economic report data provided'}, is_valid=False)
        report_data = data['report']
        if not isinstance(report_data, dict):
            return AnalysisResult(analyzer_name=self.name, result_data={
                'error': 'Invalid report data format'}, is_valid=False)
        title = report_data.get('title', '')
        content = report_data.get('content', '')
        timestamp = report_data.get('timestamp', datetime.now().isoformat())
        source = report_data.get('source', '')
        if not content:
            return AnalysisResult(analyzer_name=self.name, result_data={
                'error': 'Report content is empty'}, is_valid=False)
        report_type = report_data.get('type') or self._determine_report_type(
            report_data)
        metrics = self._extract_metrics(content)
        deviations = self._calculate_deviations(metrics)
        sentiment = self._determine_sentiment(deviations, report_type)
        currency_impacts = self._assess_currency_impacts(report_type, sentiment
            )
        pair_impacts = self._assess_pair_impacts(currency_impacts)
        result = {'report_info': {'title': title, 'type': report_type,
            'timestamp': timestamp, 'source': source}, 'metrics': metrics,
            'deviations': deviations, 'sentiment': sentiment,
            'currency_impacts': currency_impacts, 'pair_impacts':
            pair_impacts, 'trading_signals': []}
        for pair, impact in pair_impacts.items():
            if abs(impact['impact_value']) >= 0.2:
                signal = {'pair': pair, 'action': 'buy' if impact[
                    'impact_value'] > 0 else 'sell', 'strength': impact[
                    'impact_strength'], 'rationale':
                    f"{report_type} report indicates {impact['direction']}",
                    'timeframe': 'short_term'}
                result['trading_signals'].append(signal)
        return AnalysisResult(analyzer_name=self.name, result_data=result,
            is_valid=True)
