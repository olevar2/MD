"""
Entity Extraction Module

This module provides advanced entity extraction capabilities for the chat interface,
identifying domain-specific entities like currency pairs, timeframes, indicators, etc.
"""
from typing import Dict, List, Any, Optional, Union
import logging
import re
from datetime import datetime, timedelta
logger = logging.getLogger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class EntityExtractor:
    """
    Advanced entity extraction for chat messages.
    
    This class provides sophisticated entity extraction capabilities,
    identifying domain-specific entities in user messages.
    """

    def __init__(self, config: Dict[str, Any]=None):
        """
        Initialize the entity extractor.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.extractors = {'CURRENCY_PAIR': self._extract_currency_pairs,
            'TIMEFRAME': self._extract_timeframes, 'INDICATOR': self.
            _extract_indicators, 'AMOUNT': self._extract_amounts, 'PRICE':
            self._extract_prices, 'DATE': self._extract_dates, 'TIME': self
            ._extract_times, 'PERCENTAGE': self._extract_percentages,
            'TERM': self._extract_trading_terms, 'PATTERN': self.
            _extract_chart_patterns}
        self._initialize_entity_data()

    def _initialize_entity_data(self):
        """Initialize data for entity extraction."""
        self.currencies = {'USD': ['dollar', 'usd', '$', 'us dollar',
            'greenback'], 'EUR': ['euro', 'eur', '€', 'single currency'],
            'GBP': ['pound', 'gbp', '£', 'sterling', 'cable'], 'JPY': [
            'yen', 'jpy', '¥', 'japanese yen'], 'AUD': ['aussie', 'aud',
            'australian dollar'], 'CAD': ['loonie', 'cad',
            'canadian dollar'], 'CHF': ['swissie', 'chf', 'swiss franc'],
            'NZD': ['kiwi', 'nzd', 'new zealand dollar']}
        self.currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD',
            'USDCAD', 'USDCHF', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY',
            'AUDJPY', 'EURAUD', 'EURCHF', 'EURNZD', 'GBPAUD', 'GBPCAD',
            'GBPCHF', 'AUDCAD', 'AUDCHF', 'AUDNZD', 'CADCHF', 'CADJPY',
            'CHFJPY', 'NZDCAD', 'NZDCHF', 'NZDJPY']
        self.timeframes = {'1m': ['1m', '1 minute', '1min', 'one minute',
            '1-minute', 'm1'], '5m': ['5m', '5 minute', '5min',
            'five minute', '5-minute', 'm5'], '15m': ['15m', '15 minute',
            '15min', 'fifteen minute', '15-minute', 'm15'], '30m': ['30m',
            '30 minute', '30min', 'thirty minute', '30-minute', 'm30'],
            '1h': ['1h', '1 hour', '1hr', 'hourly', 'one hour', '1-hour',
            'h1'], '4h': ['4h', '4 hour', '4hr', 'four hour', '4-hour',
            'h4'], '1d': ['1d', 'daily', 'day', 'one day', '1-day', 'd1',
            'daily chart'], '1w': ['1w', 'weekly', 'week', 'one week',
            '1-week', 'w1', 'weekly chart'], '1M': ['1M', 'monthly',
            'month', 'one month', '1-month', 'M1', 'monthly chart']}
        self.indicators = {'RSI': ['rsi', 'relative strength index',
            'relative strength'], 'MACD': ['macd',
            'moving average convergence divergence'], 'Bollinger Bands': [
            'bollinger', 'bollinger bands', 'bands'], 'Moving Average': [
            'ma', 'moving average', 'average', 'ema', 'sma',
            'exponential moving average', 'simple moving average'],
            'Stochastic': ['stochastic', 'stoch', 'stochastic oscillator'],
            'Fibonacci': ['fibonacci', 'fib', 'fibonacci retracement',
            'fibonacci levels'], 'Ichimoku': ['ichimoku', 'ichimoku cloud',
            'cloud'], 'ATR': ['atr', 'average true range'], 'ADX': ['adx',
            'average directional index', 'directional index'], 'OBV': [
            'obv', 'on balance volume'], 'VWAP': ['vwap',
            'volume weighted average price']}
        self.chart_patterns = {'Head and Shoulders': ['head and shoulders',
            'h&s', 'head & shoulders', 'inverse head and shoulders'],
            'Double Top': ['double top', 'double bottom', 'double tops',
            'double bottoms'], 'Triangle': ['triangle',
            'ascending triangle', 'descending triangle',
            'symmetrical triangle'], 'Flag': ['flag', 'pennant',
            'flag pattern', 'pennant pattern'], 'Channel': ['channel',
            'ascending channel', 'descending channel', 'horizontal channel'
            ], 'Wedge': ['wedge', 'rising wedge', 'falling wedge'],
            'Cup and Handle': ['cup and handle', 'cup & handle',
            'cup with handle'], 'Engulfing': ['engulfing',
            'bullish engulfing', 'bearish engulfing'], 'Doji': ['doji',
            'doji star', 'long-legged doji', 'dragonfly doji',
            'gravestone doji']}
        self.trading_terms = {'Pip': ['pip', 'pips', 'point', 'points'],
            'Lot': ['lot', 'lots', 'micro lot', 'mini lot', 'standard lot'],
            'Spread': ['spread', 'spreads', 'bid-ask spread'], 'Leverage':
            ['leverage', 'margin', 'leveraged'], 'Stop Loss': ['stop loss',
            'sl', 'stop', 'stops'], 'Take Profit': ['take profit', 'tp',
            'profit target', 'target'], 'Trend': ['trend', 'trending',
            'uptrend', 'downtrend', 'sideways'], 'Support': ['support',
            'support level', 'support area', 'support zone'], 'Resistance':
            ['resistance', 'resistance level', 'resistance area',
            'resistance zone'], 'Breakout': ['breakout', 'break out',
            'breaking out', 'break'], 'Reversal': ['reversal', 'reverse',
            'reversing', 'trend reversal'], 'Consolidation': [
            'consolidation', 'consolidating', 'consolidate', 'range',
            'ranging']}

    def extract_entities(self, message: str) ->List[Dict[str, Any]]:
        """
        Extract entities from a message.
        
        Args:
            message: User message
            
        Returns:
            List of extracted entities
        """
        entities = []
        for entity_type, extractor in self.extractors.items():
            extracted = extractor(message)
            entities.extend(extracted)
        entities.sort(key=lambda x: x['start'])
        return entities

    def _extract_currency_pairs(self, message: str) ->List[Dict[str, Any]]:
        """Extract currency pairs from message."""
        entities = []
        message_upper = message.upper()
        for pair in self.currency_pairs:
            if pair in message_upper:
                start = message_upper.find(pair)
                entities.append({'text': pair, 'label': 'CURRENCY_PAIR',
                    'start': start, 'end': start + len(pair), 'value': pair})
        for pair in self.currency_pairs:
            base = pair[:3]
            quote = pair[3:]
            for separator in ['/', '-', ' ']:
                pair_with_separator = f'{base}{separator}{quote}'
                if pair_with_separator in message_upper:
                    start = message_upper.find(pair_with_separator)
                    entities.append({'text': pair_with_separator, 'label':
                        'CURRENCY_PAIR', 'start': start, 'end': start + len
                        (pair_with_separator), 'value': pair})
        message_lower = message.lower()
        for pair in self.currency_pairs:
            base = pair[:3]
            quote = pair[3:]
            if base in self.currencies and quote in self.currencies:
                base_aliases = self.currencies[base]
                quote_aliases = self.currencies[quote]
                for base_alias in base_aliases:
                    if base_alias in message_lower:
                        base_pos = message_lower.find(base_alias)
                        for quote_alias in quote_aliases:
                            if quote_alias in message_lower:
                                quote_pos = message_lower.find(quote_alias)
                                if abs(base_pos - quote_pos) < 20:
                                    start = min(base_pos, quote_pos)
                                    end = max(base_pos + len(base_alias), 
                                        quote_pos + len(quote_alias))
                                    entities.append({'text': message[start:
                                        end], 'label': 'CURRENCY_PAIR',
                                        'start': start, 'end': end, 'value':
                                        pair})
        return entities

    def _extract_timeframes(self, message: str) ->List[Dict[str, Any]]:
        """Extract timeframes from message."""
        entities = []
        message_lower = message.lower()
        for tf, aliases in self.timeframes.items():
            for alias in aliases:
                if alias in message_lower:
                    start = message_lower.find(alias)
                    entities.append({'text': alias, 'label': 'TIMEFRAME',
                        'start': start, 'end': start + len(alias), 'value': tf}
                        )
        numeric_timeframe_patterns = ['(\\d+)\\s*(?:minute|min|m)s?',
            '(\\d+)\\s*(?:hour|hr|h)s?', '(\\d+)\\s*(?:day|d)s?',
            '(\\d+)\\s*(?:week|w)s?', '(\\d+)\\s*(?:month|M)s?']
        for pattern in numeric_timeframe_patterns:
            matches = re.finditer(pattern, message_lower)
            for match in matches:
                value = match.group(1)
                unit = match.group(0)[len(value):].strip()
                if 'minute' in unit or 'min' in unit or unit.endswith('m'):
                    if value == '1':
                        tf_value = '1m'
                    elif value == '5':
                        tf_value = '5m'
                    elif value == '15':
                        tf_value = '15m'
                    elif value == '30':
                        tf_value = '30m'
                    else:
                        tf_value = f'{value}m'
                elif 'hour' in unit or 'hr' in unit or unit.endswith('h'):
                    if value == '1':
                        tf_value = '1h'
                    elif value == '4':
                        tf_value = '4h'
                    else:
                        tf_value = f'{value}h'
                elif 'day' in unit or unit.endswith('d'):
                    tf_value = '1d' if value == '1' else f'{value}d'
                elif 'week' in unit or unit.endswith('w'):
                    tf_value = '1w' if value == '1' else f'{value}w'
                elif 'month' in unit or unit.endswith('M'):
                    tf_value = '1M' if value == '1' else f'{value}M'
                else:
                    continue
                entities.append({'text': match.group(0), 'label':
                    'TIMEFRAME', 'start': match.start(), 'end': match.end(),
                    'value': tf_value})
        return entities

    def _extract_indicators(self, message: str) ->List[Dict[str, Any]]:
        """Extract technical indicators from message."""
        entities = []
        message_lower = message.lower()
        for indicator, aliases in self.indicators.items():
            for alias in aliases:
                if alias in message_lower:
                    start = message_lower.find(alias)
                    entities.append({'text': alias, 'label': 'INDICATOR',
                        'start': start, 'end': start + len(alias), 'value':
                        indicator})
        indicator_param_patterns = [
            '(\\d+)(?:-|\\s)?(?:period|day|bar)s?\\s+(?:rsi|macd|ma|ema|sma|stochastic)'
            ,
            '(?:rsi|macd|ma|ema|sma|stochastic)\\s+(?:with\\s+)?(\\d+)(?:-|\\s)?(?:period|day|bar)s?'
            ]
        for pattern in indicator_param_patterns:
            matches = re.finditer(pattern, message_lower)
            for match in matches:
                period = match.group(1)
                indicator_text = match.group(0)
                indicator_type = None
                for ind, aliases in self.indicators.items():
                    if any(alias in indicator_text for alias in aliases):
                        indicator_type = ind
                        break
                if indicator_type:
                    entities.append({'text': indicator_text, 'label':
                        'INDICATOR', 'start': match.start(), 'end': match.
                        end(), 'value': indicator_type, 'parameters': {
                        'period': int(period)}})
        return entities

    def _extract_amounts(self, message: str) ->List[Dict[str, Any]]:
        """Extract trading amounts from message."""
        entities = []
        amount_patterns = ['(\\d+\\.?\\d*)\\s*(lot|lots)',
            '(\\d+\\.?\\d*)\\s*(micro|mini|standard)\\s*(lot|lots)',
            '(\\d+\\.?\\d*)\\s*k', '(\\d+\\.?\\d*)\\s*units']
        for pattern in amount_patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                value = float(match.group(1))
                if 'micro' in match.group(0):
                    value *= 0.01
                elif 'mini' in match.group(0):
                    value *= 0.1
                entities.append({'text': match.group(0), 'label': 'AMOUNT',
                    'start': match.start(), 'end': match.end(), 'value': value}
                    )
        return entities

    def _extract_prices(self, message: str) ->List[Dict[str, Any]]:
        """Extract price levels from message."""
        entities = []
        price_patterns = ['at\\s+(\\d+\\.?\\d*)',
            'price\\s+of\\s+(\\d+\\.?\\d*)',
            'level\\s+of\\s+(\\d+\\.?\\d*)', '(\\d+\\.?\\d*)\\s+level',
            '(\\d+\\.?\\d*)\\s+price', '(\\d+\\.?\\d*)\\s+resistance',
            '(\\d+\\.?\\d*)\\s+support',
            'resistance\\s+at\\s+(\\d+\\.?\\d*)',
            'support\\s+at\\s+(\\d+\\.?\\d*)']
        for pattern in price_patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                price_value = float(match.group(1))
                level_type = None
                if 'resistance' in match.group(0):
                    level_type = 'resistance'
                elif 'support' in match.group(0):
                    level_type = 'support'
                entity = {'text': match.group(0), 'label': 'PRICE', 'start':
                    match.start(), 'end': match.end(), 'value': price_value}
                if level_type:
                    entity['level_type'] = level_type
                entities.append(entity)
        return entities

    @with_exception_handling
    def _extract_dates(self, message: str) ->List[Dict[str, Any]]:
        """Extract dates from message."""
        entities = []
        date_patterns = ['(\\d{4})-(\\d{1,2})-(\\d{1,2})',
            '(\\d{1,2})/(\\d{1,2})/(\\d{4})', '(\\d{1,2})-(\\d{1,2})-(\\d{4})']
        for pattern in date_patterns:
            matches = re.finditer(pattern, message)
            for match in matches:
                try:
                    if '-' in match.group(0) and len(match.group(1)) == 4:
                        year = int(match.group(1))
                        month = int(match.group(2))
                        day = int(match.group(3))
                    else:
                        month = int(match.group(1))
                        day = int(match.group(2))
                        year = int(match.group(3))
                    date_value = datetime(year, month, day).strftime('%Y-%m-%d'
                        )
                    entities.append({'text': match.group(0), 'label':
                        'DATE', 'start': match.start(), 'end': match.end(),
                        'value': date_value})
                except ValueError:
                    pass
        relative_date_patterns = {'\\btoday\\b': lambda : datetime.now().
            strftime('%Y-%m-%d'), '\\byesterday\\b': lambda : (datetime.now
            () - timedelta(days=1)).strftime('%Y-%m-%d'), '\\btomorrow\\b':
            lambda : (datetime.now() + timedelta(days=1)).strftime(
            '%Y-%m-%d'), '\\bnext\\s+week\\b': lambda : (datetime.now() +
            timedelta(weeks=1)).strftime('%Y-%m-%d'), '\\blast\\s+week\\b':
            lambda : (datetime.now() - timedelta(weeks=1)).strftime(
            '%Y-%m-%d'), '\\bnext\\s+month\\b': lambda : (datetime.now() +
            timedelta(days=30)).strftime('%Y-%m-%d'), '\\blast\\s+month\\b':
            lambda : (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            }
        for pattern, date_func in relative_date_patterns.items():
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                entities.append({'text': match.group(0), 'label': 'DATE',
                    'start': match.start(), 'end': match.end(), 'value':
                    date_func()})
        return entities

    @with_exception_handling
    def _extract_times(self, message: str) ->List[Dict[str, Any]]:
        """Extract times from message."""
        entities = []
        time_patterns = ['(\\d{1,2}):(\\d{2})(?::(\\d{2}))?\\s*(am|pm)?',
            '(\\d{1,2})\\s*(am|pm)']
        for pattern in time_patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                try:
                    if ':' in match.group(0):
                        hour = int(match.group(1))
                        minute = int(match.group(2))
                        second = int(match.group(3)) if match.group(3) else 0
                        if match.group(4):
                            if match.group(4).lower() == 'pm' and hour < 12:
                                hour += 12
                            elif match.group(4).lower() == 'am' and hour == 12:
                                hour = 0
                    else:
                        hour = int(match.group(1))
                        minute = 0
                        second = 0
                        if match.group(2).lower() == 'pm' and hour < 12:
                            hour += 12
                        elif match.group(2).lower() == 'am' and hour == 12:
                            hour = 0
                    time_value = f'{hour:02d}:{minute:02d}:{second:02d}'
                    entities.append({'text': match.group(0), 'label':
                        'TIME', 'start': match.start(), 'end': match.end(),
                        'value': time_value})
                except (ValueError, IndexError):
                    pass
        return entities

    def _extract_percentages(self, message: str) ->List[Dict[str, Any]]:
        """Extract percentages from message."""
        entities = []
        percentage_patterns = ['(\\d+\\.?\\d*)%',
            '(\\d+\\.?\\d*)\\s+percent', '(\\d+\\.?\\d*)\\s+pct']
        for pattern in percentage_patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                value = float(match.group(1))
                entities.append({'text': match.group(0), 'label':
                    'PERCENTAGE', 'start': match.start(), 'end': match.end(
                    ), 'value': value})
        return entities

    def _extract_trading_terms(self, message: str) ->List[Dict[str, Any]]:
        """Extract trading terms from message."""
        entities = []
        message_lower = message.lower()
        for term, aliases in self.trading_terms.items():
            for alias in aliases:
                if alias in message_lower:
                    start = message_lower.find(alias)
                    entities.append({'text': alias, 'label': 'TERM',
                        'start': start, 'end': start + len(alias), 'value':
                        term})
        return entities

    def _extract_chart_patterns(self, message: str) ->List[Dict[str, Any]]:
        """Extract chart patterns from message."""
        entities = []
        message_lower = message.lower()
        for pattern, aliases in self.chart_patterns.items():
            for alias in aliases:
                if alias in message_lower:
                    start = message_lower.find(alias)
                    entities.append({'text': alias, 'label': 'PATTERN',
                        'start': start, 'end': start + len(alias), 'value':
                        pattern})
        return entities
