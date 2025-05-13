"""
Historical News Data Collector for Forex Trading Platform.

This module provides utilities to collect, parse, and format historical 
news events data for use in the news-aware backtesting framework.
"""
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import json
import os
import logging
from bs4 import BeautifulSoup
import csv
import re
from core_foundations.utils.logger import get_logger
from trading_gateway_service.simulation.news_sentiment_simulator import NewsEvent, NewsImpactLevel, NewsEventType, SentimentLevel
logger = get_logger(__name__)
from trading_gateway_service.error.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from trading_gateway_service.resilience.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class NewsDataCollector:
    """
    Collector and parser for historical forex news events.
    
    This class handles collecting historical news data from various sources,
    parsing it, and preparing it in a format suitable for backtesting.
    """

    def __init__(self, cache_dir: str=None):
        """
        Initialize the news data collector.
        
        Args:
            cache_dir: Directory to cache collected news data
        """
        self.cache_dir = cache_dir or os.path.join('data', 'historical_news')
        os.makedirs(self.cache_dir, exist_ok=True)

    @with_exception_handling
    def parse_csv_news_data(self, csv_path: str) ->List[NewsEvent]:
        """
        Parse historical news data from CSV file.
        
        Expected CSV format:
        date,time,currency,importance,event,actual,forecast,previous
        
        Args:
            csv_path: Path to CSV file with news data
            
        Returns:
            List of parsed NewsEvent objects
        """
        logger.info(f'Parsing news data from {csv_path}')
        events = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    try:
                        date_str = row.get('date', '')
                        time_str = row.get('time', '00:00')
                        try:
                            if '-' in date_str:
                                date_obj = datetime.strptime(date_str,
                                    '%Y-%m-%d')
                            elif '/' in date_str:
                                date_obj = datetime.strptime(date_str,
                                    '%m/%d/%Y')
                            else:
                                continue
                        except ValueError:
                            logger.warning(f'Could not parse date: {date_str}')
                            continue
                        try:
                            time_parts = time_str.split(':')
                            date_obj = date_obj.replace(hour=int(time_parts
                                [0]), minute=int(time_parts[1]))
                        except (ValueError, IndexError):
                            logger.warning(f'Could not parse time: {time_str}')
                        importance = row.get('importance', '').lower()
                        if 'high' in importance or 'red' in importance:
                            impact_level = NewsImpactLevel.HIGH
                        elif 'medium' in importance or 'orange' in importance:
                            impact_level = NewsImpactLevel.MEDIUM
                        elif 'low' in importance or 'yellow' in importance:
                            impact_level = NewsImpactLevel.LOW
                        else:
                            impact_level = NewsImpactLevel.MEDIUM
                        currency = row.get('currency', '')
                        if not currency:
                            continue
                        pairs = self._currency_to_pairs(currency)
                        if not pairs:
                            continue
                        actual = self._parse_value(row.get('actual', ''))
                        forecast = self._parse_value(row.get('forecast', ''))
                        previous = self._parse_value(row.get('previous', ''))
                        event_title = row.get('event', '')
                        event_type = self._determine_event_type(event_title)
                        price_impact, volatility_impact = (self.
                            _calculate_impact(actual, forecast, previous,
                            impact_level))
                        sentiment_impact = self._determine_sentiment_impact(
                            actual, forecast, previous, event_title)
                        event = NewsEvent(event_id=f'hist_{i}', event_type=
                            event_type, impact_level=impact_level,
                            timestamp=date_obj, currencies_affected=pairs,
                            title=event_title, description=
                            f'Historical {event_title}', expected_value=
                            forecast, actual_value=actual, previous_value=
                            previous, sentiment_impact=sentiment_impact,
                            volatility_impact=volatility_impact,
                            price_impact=price_impact, duration_minutes=
                            self._get_duration_for_impact(impact_level))
                        events.append(event)
                    except Exception as e:
                        logger.error(f'Error parsing news event: {str(e)}')
                        continue
            logger.info(f'Successfully parsed {len(events)} news events')
            return events
        except Exception as e:
            logger.error(f'Error reading news CSV file: {str(e)}')
            return []

    @with_database_resilience('load_or_download_news_for_period')
    @with_exception_handling
    def load_or_download_news_for_period(self, start_date: datetime,
        end_date: datetime, source: str='local', force_reload: bool=False
        ) ->List[NewsEvent]:
        """
        Load news events for a period, downloading if necessary.
        
        Args:
            start_date: Start of period
            end_date: End of period
            source: Data source ('local', 'forexfactory', etc.)
            force_reload: Force reload even if cached data exists
            
        Returns:
            List of NewsEvent objects
        """
        cache_filename = (
            f"news_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
            )
        cache_path = os.path.join(self.cache_dir, cache_filename)
        if os.path.exists(cache_path) and not force_reload:
            logger.info(f'Loading cached news data from {cache_path}')
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                events = []
                for event_dict in data:
                    events.append(NewsEvent.from_dict(event_dict))
                logger.info(f'Loaded {len(events)} cached news events')
                return events
            except Exception as e:
                logger.error(f'Error loading cached news: {str(e)}')
        events = []
        if source == 'local':
            year_months = set()
            current_date = start_date
            while current_date <= end_date:
                year_months.add((current_date.year, current_date.month))
                current_date += timedelta(days=1)
            for year, month in year_months:
                csv_path = os.path.join(self.cache_dir,
                    f'{year}_{month:02d}.csv')
                if os.path.exists(csv_path):
                    logger.info(f'Found local news data: {csv_path}')
                    events.extend(self.parse_csv_news_data(csv_path))
        elif source == 'sample':
            logger.info('Generating synthetic news data for testing')
            days_range = (end_date - start_date).days
            if days_range <= 0:
                days_range = 1
            major_events = []
            for i in range(min(days_range // 7 + 1, 5)):
                event_date = start_date + timedelta(days=i * 7 + np.random.
                    randint(0, 7))
                hour = np.random.randint(8, 16)
                event_date = event_date.replace(hour=hour, minute=0)
                if i % 2 == 0:
                    event = NewsEvent(event_id=f'sample_major_{i}',
                        event_type=NewsEventType.ECONOMIC_DATA,
                        impact_level=NewsImpactLevel.HIGH, timestamp=
                        event_date, currencies_affected=['EUR/USD',
                        'GBP/USD', 'USD/JPY'], title='US Non-Farm Payrolls',
                        description='Major employment report',
                        expected_value=200.0, actual_value=180.0,
                        previous_value=190.0, sentiment_impact=
                        SentimentLevel.SLIGHTLY_BEARISH, volatility_impact=
                        2.5, price_impact=-0.002, duration_minutes=180)
                else:
                    event = NewsEvent(event_id=f'sample_major_{i}',
                        event_type=NewsEventType.CENTRAL_BANK, impact_level
                        =NewsImpactLevel.CRITICAL, timestamp=event_date,
                        currencies_affected=['EUR/USD', 'EUR/GBP',
                        'EUR/JPY'], title='ECB Interest Rate Decision',
                        description='European Central Bank policy meeting',
                        expected_value=0.25, actual_value=0.25,
                        previous_value=0.25, sentiment_impact=
                        SentimentLevel.NEUTRAL, volatility_impact=3.0,
                        price_impact=0.001, duration_minutes=240)
                major_events.append(event)
            medium_events = []
            for i in range(min(days_range, 10)):
                event_date = start_date + timedelta(days=i * days_range // 10)
                hour = np.random.randint(8, 16)
                event_date = event_date.replace(hour=hour, minute=0)
                event = NewsEvent(event_id=f'sample_medium_{i}', event_type
                    =NewsEventType.ECONOMIC_DATA, impact_level=
                    NewsImpactLevel.MEDIUM, timestamp=event_date,
                    currencies_affected=['GBP/USD'], title=
                    'UK Manufacturing PMI', description=
                    'Purchasing Managers Index', expected_value=52.0,
                    actual_value=53.5, previous_value=51.8,
                    sentiment_impact=SentimentLevel.SLIGHTLY_BULLISH,
                    volatility_impact=1.5, price_impact=0.0015,
                    duration_minutes=120)
                medium_events.append(event)
            events = major_events + medium_events
        if events:
            try:
                with open(cache_path, 'w') as f:
                    json.dump([e.to_dict() for e in events], f, indent=2,
                        default=str)
                logger.info(f'Cached {len(events)} news events to {cache_path}'
                    )
            except Exception as e:
                logger.error(f'Error caching news data: {str(e)}')
        return events

    @with_exception_handling
    def _parse_value(self, value_str: str) ->Optional[float]:
        """Parse a value from a string, handling various formats."""
        if not value_str or value_str.strip() in ['-', '', 'n/a', 'N/A']:
            return None
        value_str = re.sub('[^\\d.-]', '', value_str)
        try:
            return float(value_str)
        except (ValueError, TypeError):
            return None

    def _currency_to_pairs(self, currency: str) ->List[str]:
        """Convert a currency code to a list of relevant pairs."""
        major_pairs = {'USD': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF',
            'AUD/USD', 'USD/CAD', 'NZD/USD'], 'EUR': ['EUR/USD', 'EUR/GBP',
            'EUR/JPY', 'EUR/CHF', 'EUR/AUD', 'EUR/CAD', 'EUR/NZD'], 'GBP':
            ['GBP/USD', 'EUR/GBP', 'GBP/JPY', 'GBP/CHF', 'GBP/AUD',
            'GBP/CAD', 'GBP/NZD'], 'JPY': ['USD/JPY', 'EUR/JPY', 'GBP/JPY',
            'CHF/JPY', 'AUD/JPY', 'CAD/JPY', 'NZD/JPY'], 'AUD': ['AUD/USD',
            'EUR/AUD', 'GBP/AUD', 'AUD/JPY', 'AUD/CHF', 'AUD/CAD',
            'AUD/NZD'], 'CAD': ['USD/CAD', 'EUR/CAD', 'GBP/CAD', 'CAD/JPY',
            'AUD/CAD', 'CAD/CHF', 'NZD/CAD'], 'CHF': ['USD/CHF', 'EUR/CHF',
            'GBP/CHF', 'CHF/JPY', 'AUD/CHF', 'CAD/CHF', 'NZD/CHF'], 'NZD':
            ['NZD/USD', 'EUR/NZD', 'GBP/NZD', 'NZD/JPY', 'AUD/NZD',
            'NZD/CAD', 'NZD/CHF']}
        currency = currency.strip().upper()
        if 'EURO' in currency or 'EUROPEAN' in currency:
            currency = 'EUR'
        elif 'STERLING' in currency or 'BRITISH' in currency or 'UK' in currency:
            currency = 'GBP'
        elif 'AUSSIE' in currency or 'AUSTRALIAN' in currency:
            currency = 'AUD'
        elif 'KIWI' in currency or 'NEW ZEALAND' in currency:
            currency = 'NZD'
        elif 'YEN' in currency or 'JAPANESE' in currency:
            currency = 'JPY'
        elif 'LOONIE' in currency or 'CANADIAN' in currency:
            currency = 'CAD'
        elif 'SWISSIE' in currency or 'SWISS' in currency:
            currency = 'CHF'
        elif 'GREENBACK' in currency or 'US ' in currency or 'U.S.' in currency or 'AMERICAN' in currency:
            currency = 'USD'
        match = re.search('([A-Z]{3})', currency)
        if match:
            currency = match.group(1)
        return major_pairs.get(currency, [])

    def _determine_event_type(self, event_title: str) ->NewsEventType:
        """Determine the event type from its title."""
        title_lower = event_title.lower()
        if any(x in title_lower for x in ['rate', 'central bank', 'fed',
            'ecb', 'boe', 'rba', 'boj', 'fomc', 'minutes']):
            return NewsEventType.CENTRAL_BANK
        elif any(x in title_lower for x in ['election', 'vote',
            'referendum', 'war', 'conflict', 'tariff', 'trade war']):
            return NewsEventType.GEOPOLITICAL
        elif any(x in title_lower for x in ['earthquake', 'hurricane',
            'tsunami', 'volcano', 'flood', 'disaster']):
            return NewsEventType.NATURAL_DISASTER
        elif any(x in title_lower for x in ['sentiment', 'confidence',
            'outlook', 'survey']):
            return NewsEventType.MARKET_SENTIMENT
        else:
            return NewsEventType.ECONOMIC_DATA

    def _calculate_impact(self, actual: Optional[float], forecast: Optional
        [float], previous: Optional[float], impact_level: NewsImpactLevel
        ) ->Tuple[float, float]:
        """Calculate price and volatility impact based on data values."""
        base_factors = {NewsImpactLevel.LOW: 0.0005, NewsImpactLevel.MEDIUM:
            0.001, NewsImpactLevel.HIGH: 0.002, NewsImpactLevel.CRITICAL: 0.005
            }
        base_volatility = {NewsImpactLevel.LOW: 1.2, NewsImpactLevel.MEDIUM:
            1.5, NewsImpactLevel.HIGH: 2.0, NewsImpactLevel.CRITICAL: 3.0}
        base_factor = base_factors[impact_level]
        volatility = base_volatility[impact_level]
        if actual is not None and forecast is not None and forecast != 0:
            surprise_factor = (actual - forecast) / abs(forecast)
            surprise_factor = max(min(surprise_factor, 1.0), -1.0)
            price_impact = base_factor * surprise_factor
            volatility_impact = volatility * (1 + abs(surprise_factor))
        else:
            price_impact = np.random.normal(0, base_factor)
            volatility_impact = volatility
        volatility_impact = max(1.01, volatility_impact)
        return price_impact, volatility_impact

    def _determine_sentiment_impact(self, actual: Optional[float], forecast:
        Optional[float], previous: Optional[float], event_title: str
        ) ->SentimentLevel:
        """Determine sentiment impact based on data values and event type."""
        if actual is None or forecast is None:
            return SentimentLevel.NEUTRAL
        if forecast != 0:
            surprise_factor = (actual - forecast) / abs(forecast)
        else:
            surprise_factor = 0
        title_lower = event_title.lower()
        if any(x in title_lower for x in ['unemployment', 'jobless',
            'deficit', 'debt', 'inflation']):
            surprise_factor = -surprise_factor
        if surprise_factor < -0.5:
            return SentimentLevel.VERY_BEARISH
        elif surprise_factor < -0.25:
            return SentimentLevel.BEARISH
        elif surprise_factor < -0.1:
            return SentimentLevel.SLIGHTLY_BEARISH
        elif surprise_factor <= 0.1:
            return SentimentLevel.NEUTRAL
        elif surprise_factor <= 0.25:
            return SentimentLevel.SLIGHTLY_BULLISH
        elif surprise_factor <= 0.5:
            return SentimentLevel.BULLISH
        else:
            return SentimentLevel.VERY_BULLISH

    def _get_duration_for_impact(self, impact_level: NewsImpactLevel) ->int:
        """Get a suitable duration in minutes based on impact level."""
        durations = {NewsImpactLevel.LOW: 60, NewsImpactLevel.MEDIUM: 120,
            NewsImpactLevel.HIGH: 240, NewsImpactLevel.CRITICAL: 480}
        return durations.get(impact_level, 120)


@with_exception_handling
def convert_csv_format(input_path, output_path):
    """
    Convert a simple CSV format to the expected format.
    
    Args:
        input_path: Path to input CSV
        output_path: Path to save the formatted CSV
    """
    try:
        df = pd.read_csv(input_path)
        required_cols = ['date', 'time', 'currency', 'importance', 'event']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            col_map = {}
            for col in df.columns:
                if 'date' in col.lower():
                    col_map[col] = 'date'
                elif 'time' in col.lower():
                    col_map[col] = 'time'
                elif 'curr' in col.lower():
                    col_map[col] = 'currency'
                elif 'imp' in col.lower() or 'impact' in col.lower():
                    col_map[col] = 'importance'
                elif 'event' in col.lower() or 'title' in col.lower(
                    ) or 'name' in col.lower():
                    col_map[col] = 'event'
                elif 'actual' in col.lower():
                    col_map[col] = 'actual'
                elif 'forecast' in col.lower() or 'expected' in col.lower():
                    col_map[col] = 'forecast'
                elif 'previous' in col.lower() or 'prev' in col.lower():
                    col_map[col] = 'previous'
            if col_map:
                df = df.rename(columns=col_map)
        for col in required_cols:
            if col not in df.columns:
                df[col] = ''
        for col in ['actual', 'forecast', 'previous']:
            if col not in df.columns:
                df[col] = np.nan
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            except:
                pass
        df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        logger.error(f'Error converting CSV format: {str(e)}')
        return False
