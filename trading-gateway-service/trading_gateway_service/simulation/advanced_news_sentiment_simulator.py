"""
News and Sentiment Simulator for Forex Trading Platform.

This module provides realistic simulation of news events and market sentiment,
including their impact on price movements, volatility, and liquidity.
"""
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
import random
import logging
import uuid
import json
from core_foundations.utils.logger import get_logger
from trading_gateway_service.simulation.forex_broker_simulator import ForexBrokerSimulator
logger = get_logger(__name__)


from trading_gateway_service.resilience.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class NewsCategory(str, Enum):
    """Categories of news events."""
    ECONOMIC_DATA = 'economic_data'
    CENTRAL_BANK = 'central_bank'
    GEOPOLITICAL = 'geopolitical'
    CORPORATE = 'corporate'
    MARKET_SENTIMENT = 'market_sentiment'
    NATURAL_DISASTER = 'natural_disaster'
    REGULATORY = 'regulatory'
    COMMODITY = 'commodity'


class NewsImpactLevel(str, Enum):
    """Impact level of news events."""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'


class SentimentLevel(str, Enum):
    """Market sentiment levels."""
    VERY_BEARISH = 'very_bearish'
    BEARISH = 'bearish'
    SLIGHTLY_BEARISH = 'slightly_bearish'
    NEUTRAL = 'neutral'
    SLIGHTLY_BULLISH = 'slightly_bullish'
    BULLISH = 'bullish'
    VERY_BULLISH = 'very_bullish'


class TechnicalImpact(str, Enum):
    """Technical impact of news events on market conditions."""
    TREND_REVERSAL = 'trend_reversal'
    TREND_ACCELERATION = 'trend_acceleration'
    VOLATILITY_SPIKE = 'volatility_spike'
    LIQUIDITY_REDUCTION = 'liquidity_reduction'
    RANGE_BREAKOUT = 'range_breakout'
    GAP_EVENT = 'gap_event'
    NONE = 'none'


class NewsEvent:
    """
    Represents a news event that can impact the forex market.
    """

    def __init__(self, event_id: str, category: NewsCategory, impact_level:
        NewsImpactLevel, title: str, description: str, release_time:
        datetime, affected_currencies: List[str], expected_value: Optional[
        str]=None, actual_value: Optional[str]=None, previous_value:
        Optional[str]=None, sentiment_impact: Dict[str, SentimentLevel]=
        None, technical_impact: TechnicalImpact=TechnicalImpact.NONE,
        price_impact_factor: float=0.0, volatility_impact_factor: float=1.0,
        liquidity_impact_factor: float=1.0, duration_minutes: int=60):
        """
        Initialize a news event.
        
        Args:
            event_id: Unique identifier for the event
            category: Category of the news event
            impact_level: Impact level of the event
            title: Title of the news event
            description: Description of the event
            release_time: When the event is released
            affected_currencies: List of currency codes affected
            expected_value: Expected economic data value if applicable
            actual_value: Actual economic data value if applicable
            previous_value: Previous economic data value if applicable
            sentiment_impact: Sentiment impact by currency
            technical_impact: Technical impact on charts
            price_impact_factor: Direct price movement factor
            volatility_impact_factor: Impact on volatility (1.0 = normal)
            liquidity_impact_factor: Impact on liquidity (1.0 = normal)
            duration_minutes: How long the impact lasts
        """
        self.event_id = event_id
        self.category = category
        self.impact_level = impact_level
        self.title = title
        self.description = description
        self.release_time = release_time
        self.affected_currencies = affected_currencies
        self.expected_value = expected_value
        self.actual_value = actual_value
        self.previous_value = previous_value
        self.sentiment_impact = sentiment_impact or {}
        self.technical_impact = technical_impact
        self.price_impact_factor = price_impact_factor
        self.volatility_impact_factor = volatility_impact_factor
        self.liquidity_impact_factor = liquidity_impact_factor
        self.duration_minutes = duration_minutes
        self.processed = False

    @with_market_data_resilience('get_price_impact_factor')
    def get_price_impact_factor(self) ->float:
        """Get the price impact factor for this event."""
        return self.price_impact_factor

    @with_broker_api_resilience('get_volatility_impact_factor')
    def get_volatility_impact_factor(self) ->float:
        """Get the volatility impact factor for this event."""
        return self.volatility_impact_factor

    @with_broker_api_resilience('get_liquidity_impact_factor')
    def get_liquidity_impact_factor(self) ->float:
        """Get the liquidity impact factor for this event."""
        return self.liquidity_impact_factor

    @with_broker_api_resilience('calculate_remaining_impact')
    def calculate_remaining_impact(self, current_time: datetime) ->float:
        """
        Calculate the remaining impact factor based on time elapsed.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Remaining impact factor (0.0-1.0)
        """
        if not self.processed:
            return 0.0
        end_time = self.release_time + timedelta(minutes=self.duration_minutes)
        if current_time < self.release_time:
            return 0.0
        elif current_time > end_time:
            return 0.0
        else:
            elapsed = (current_time - self.release_time).total_seconds() / 60.0
            remaining_pct = 1.0 - min(1.0, elapsed / max(1, self.
                duration_minutes))
            if elapsed < self.duration_minutes * 0.1:
                rise_pct = elapsed / (self.duration_minutes * 0.1)
                return rise_pct
            else:
                return remaining_pct

    def to_dict(self) ->Dict[str, Any]:
        """Convert event to dictionary."""
        return {'event_id': self.event_id, 'category': self.category.value,
            'impact_level': self.impact_level.value, 'title': self.title,
            'description': self.description, 'release_time': self.
            release_time.isoformat(), 'affected_currencies': self.
            affected_currencies, 'expected_value': self.expected_value,
            'actual_value': self.actual_value, 'previous_value': self.
            previous_value, 'sentiment_impact': {k: v.value for k, v in
            self.sentiment_impact.items()}, 'technical_impact': self.
            technical_impact.value, 'price_impact_factor': self.
            price_impact_factor, 'volatility_impact_factor': self.
            volatility_impact_factor, 'liquidity_impact_factor': self.
            liquidity_impact_factor, 'duration_minutes': self.
            duration_minutes, 'processed': self.processed}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) ->'NewsEvent':
        """Create event from dictionary."""
        return cls(event_id=data['event_id'], category=NewsCategory(data[
            'category']), impact_level=NewsImpactLevel(data['impact_level']
            ), title=data['title'], description=data['description'],
            release_time=datetime.fromisoformat(data['release_time']),
            affected_currencies=data['affected_currencies'], expected_value
            =data.get('expected_value'), actual_value=data.get(
            'actual_value'), previous_value=data.get('previous_value'),
            sentiment_impact={k: SentimentLevel(v) for k, v in data.get(
            'sentiment_impact', {}).items()}, technical_impact=
            TechnicalImpact(data.get('technical_impact', 'none')),
            price_impact_factor=data.get('price_impact_factor', 0.0),
            volatility_impact_factor=data.get('volatility_impact_factor', 
            1.0), liquidity_impact_factor=data.get(
            'liquidity_impact_factor', 1.0), duration_minutes=data.get(
            'duration_minutes', 60))


class NewsCalendar:
    """
    Calendar of scheduled and unscheduled news events.
    """

    def __init__(self, events: Optional[List[NewsEvent]]=None):
        """
        Initialize the news calendar.
        
        Args:
            events: Initial list of news events
        """
        self.events = events or []
        self.events_by_id = {event.event_id: event for event in self.events}

    def add_event(self, event: NewsEvent) ->None:
        """Add an event to the calendar."""
        self.events.append(event)
        self.events_by_id[event.event_id] = event
        self.events.sort(key=lambda e: e.release_time)

    @with_broker_api_resilience('get_event')
    def get_event(self, event_id: str) ->Optional[NewsEvent]:
        """Get an event by ID."""
        return self.events_by_id.get(event_id)

    @with_broker_api_resilience('get_events_in_window')
    def get_events_in_window(self, start_time: datetime, end_time: datetime,
        category: Optional[NewsCategory]=None, impact_level: Optional[
        NewsImpactLevel]=None) ->List[NewsEvent]:
        """
        Get events within a time window with optional filtering.
        
        Args:
            start_time: Start of time window
            end_time: End of time window
            category: Optional filter by category
            impact_level: Optional filter by impact level
            
        Returns:
            List of matching events
        """
        matching_events = []
        for event in self.events:
            if start_time <= event.release_time <= end_time:
                if category and event.category != category:
                    continue
                if impact_level and event.impact_level != impact_level:
                    continue
                matching_events.append(event)
        return matching_events

    @with_broker_api_resilience('get_active_events')
    def get_active_events(self, current_time: datetime) ->List[NewsEvent]:
        """
        Get events that are currently active.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            List of active events
        """
        active_events = []
        for event in self.events:
            if not event.processed:
                continue
            end_time = event.release_time + timedelta(minutes=event.
                duration_minutes)
            if event.release_time <= current_time <= end_time:
                active_events.append(event)
        return active_events

    @with_broker_api_resilience('update_events')
    def update_events(self, current_time: datetime) ->List[NewsEvent]:
        """
        Update events based on current time, marking events as processed if their
        release time has been reached.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            List of newly processed events
        """
        newly_processed = []
        for event in self.events:
            if not event.processed and event.release_time <= current_time:
                event.processed = True
                newly_processed.append(event)
        return newly_processed

    def save_to_file(self, filepath: str) ->None:
        """
        Save calendar to a JSON file.
        
        Args:
            filepath: Path to save file
        """
        with open(filepath, 'w') as f:
            json.dump([event.to_dict() for event in self.events], f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) ->'NewsCalendar':
        """
        Load calendar from a JSON file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            NewsCalendar instance
        """
        with open(filepath, 'r') as f:
            event_dicts = json.load(f)
        events = [NewsEvent.from_dict(event_dict) for event_dict in event_dicts
            ]
        return cls(events)


class MarketSentimentTracker:
    """
    Tracks market sentiment across different currencies and pairs.
    """

    def __init__(self, base_sentiment: Dict[str, SentimentLevel]=None):
        """
        Initialize the sentiment tracker.
        
        Args:
            base_sentiment: Initial sentiment by currency code
        """
        self.base_sentiment = base_sentiment or {'USD': SentimentLevel.
            NEUTRAL, 'EUR': SentimentLevel.NEUTRAL, 'GBP': SentimentLevel.
            NEUTRAL, 'JPY': SentimentLevel.NEUTRAL, 'AUD': SentimentLevel.
            NEUTRAL, 'CAD': SentimentLevel.NEUTRAL, 'CHF': SentimentLevel.
            NEUTRAL, 'NZD': SentimentLevel.NEUTRAL}
        self.sentiment_history = {currency: [] for currency in self.
            base_sentiment}
        self.active_sentiment_impacts = []

    def sentiment_to_value(self, sentiment: SentimentLevel) ->float:
        """
        Convert sentiment level to numeric value (-1.0 to 1.0).
        
        Args:
            sentiment: Sentiment level
            
        Returns:
            Numeric sentiment value
        """
        mapping = {SentimentLevel.VERY_BEARISH: -1.0, SentimentLevel.
            BEARISH: -0.66, SentimentLevel.SLIGHTLY_BEARISH: -0.33,
            SentimentLevel.NEUTRAL: 0.0, SentimentLevel.SLIGHTLY_BULLISH: 
            0.33, SentimentLevel.BULLISH: 0.66, SentimentLevel.VERY_BULLISH:
            1.0}
        return mapping.get(sentiment, 0.0)

    def value_to_sentiment(self, value: float) ->SentimentLevel:
        """
        Convert numeric value to sentiment level.
        
        Args:
            value: Numeric sentiment value (-1.0 to 1.0)
            
        Returns:
            Sentiment level
        """
        if value <= -0.8:
            return SentimentLevel.VERY_BEARISH
        elif value <= -0.5:
            return SentimentLevel.BEARISH
        elif value <= -0.2:
            return SentimentLevel.SLIGHTLY_BEARISH
        elif value < 0.2:
            return SentimentLevel.NEUTRAL
        elif value < 0.5:
            return SentimentLevel.SLIGHTLY_BULLISH
        elif value < 0.8:
            return SentimentLevel.BULLISH
        else:
            return SentimentLevel.VERY_BULLISH

    @with_broker_api_resilience('update_sentiment')
    def update_sentiment(self, currency: str, event: NewsEvent,
        event_impact: float) ->None:
        """
        Update sentiment based on news event.
        
        Args:
            currency: Currency code
            event: News event
            event_impact: Event's current impact factor (0.0-1.0)
        """
        if currency not in self.base_sentiment:
            self.base_sentiment[currency] = SentimentLevel.NEUTRAL
            self.sentiment_history[currency] = []
        impact_sentiment = event.sentiment_impact.get(currency,
            SentimentLevel.NEUTRAL)
        impact_value = self.sentiment_to_value(impact_sentiment)
        scaled_impact = impact_value * event_impact
        self.active_sentiment_impacts.append({'currency': currency,
            'event_id': event.event_id, 'impact_value': scaled_impact,
            'expiry': datetime.now() + timedelta(minutes=event.
            duration_minutes)})

    @with_broker_api_resilience('get_current_sentiment')
    def get_current_sentiment(self, currency: str) ->SentimentLevel:
        """
        Get current sentiment for a currency.
        
        Args:
            currency: Currency code
            
        Returns:
            Current sentiment level
        """
        if currency not in self.base_sentiment:
            return SentimentLevel.NEUTRAL
        base_value = self.sentiment_to_value(self.base_sentiment[currency])
        current_time = datetime.now()
        active_impacts = [impact for impact in self.
            active_sentiment_impacts if impact['currency'] == currency and 
            impact['expiry'] > current_time]
        total_impact = sum(impact['impact_value'] for impact in active_impacts)
        final_value = base_value + total_impact
        final_value = max(-1.0, min(1.0, final_value))
        return self.value_to_sentiment(final_value)

    @with_broker_api_resilience('get_pair_sentiment')
    def get_pair_sentiment(self, base_currency: str, quote_currency: str
        ) ->SentimentLevel:
        """
        Get sentiment for a currency pair.
        
        Args:
            base_currency: Base currency code
            quote_currency: Quote currency code
            
        Returns:
            Pair sentiment level
        """
        base_sentiment = self.sentiment_to_value(self.get_current_sentiment
            (base_currency))
        quote_sentiment = self.sentiment_to_value(self.
            get_current_sentiment(quote_currency))
        pair_value = base_sentiment - quote_sentiment
        pair_value = max(-1.0, min(1.0, pair_value))
        return self.value_to_sentiment(pair_value)

    @with_broker_api_resilience('get_historical_sentiment')
    def get_historical_sentiment(self, currency: str, lookback: int=30) ->List[
        SentimentLevel]:
        """
        Get historical sentiment levels.
        
        Args:
            currency: Currency code
            lookback: Number of historical points to retrieve
            
        Returns:
            List of historical sentiment levels
        """
        if currency not in self.sentiment_history:
            return [SentimentLevel.NEUTRAL] * min(lookback, 1)
        history = self.sentiment_history[currency]
        if not history:
            return [SentimentLevel.NEUTRAL] * min(lookback, 1)
        return history[-lookback:]

    def record_current_sentiment(self) ->None:
        """Record current sentiment for all currencies in history."""
        for currency in self.base_sentiment:
            current_sentiment = self.get_current_sentiment(currency)
            self.sentiment_history[currency].append(current_sentiment)
            max_history = 1000
            if len(self.sentiment_history[currency]) > max_history:
                self.sentiment_history[currency] = self.sentiment_history[
                    currency][-max_history:]


class NewsAndSentimentSimulator:
    """
    Simulator for news events and market sentiment.
    
    This class generates and processes realistic news events and their impact on
    market conditions, including price movements, volatility, and sentiment.
    """

    def __init__(self, broker_simulator: ForexBrokerSimulator,
        news_calendar: Optional[NewsCalendar]=None, sentiment_tracker:
        Optional[MarketSentimentTracker]=None, event_frequency: float=0.05,
        random_seed: Optional[int]=None):
        """
        Initialize the news and sentiment simulator.
        
        Args:
            broker_simulator: Forex broker simulator
            news_calendar: Optional pre-configured news calendar
            sentiment_tracker: Optional pre-configured sentiment tracker
            event_frequency: Frequency of random events per hour
            random_seed: Random seed for reproducibility
        """
        self.broker_simulator = broker_simulator
        self.news_calendar = news_calendar or NewsCalendar()
        self.sentiment_tracker = sentiment_tracker or MarketSentimentTracker()
        self.event_frequency = event_frequency
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        self.current_time = datetime.now()
        self.last_update_time = self.current_time
        self.last_event_generation = self.current_time

    def update(self, current_time: datetime) ->List[NewsEvent]:
        """
        Update the simulator to the current time.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of newly processed events
        """
        self.current_time = current_time
        time_elapsed = (current_time - self.last_update_time).total_seconds(
            ) / 3600.0
        if time_elapsed > 0 and self.event_frequency > 0:
            event_prob = self.event_frequency * time_elapsed
            if random.random() < event_prob:
                self._generate_random_event(current_time)
        newly_processed = self.news_calendar.update_events(current_time)
        for event in newly_processed:
            self._apply_event_impact(event)
        self.sentiment_tracker.record_current_sentiment()
        self.last_update_time = current_time
        return newly_processed

    def _apply_event_impact(self, event: NewsEvent) ->None:
        """
        Apply the impact of a news event to the market.
        
        Args:
            event: The news event to apply
        """
        logger.info(f'Applying impact of news event: {event.title}')
        affected_symbols = self._get_affected_symbols(event.affected_currencies
            )
        for currency in event.affected_currencies:
            self.sentiment_tracker.update_sentiment(currency, event, 1.0)
        market_event = self._create_market_event_from_news(event,
            affected_symbols)
        self.broker_simulator.add_market_event(market_event)

    def _create_market_event_from_news(self, news_event: NewsEvent,
        affected_symbols: List[str]) ->Any:
        """
        Create a market event from a news event.
        
        Args:
            news_event: News event
            affected_symbols: List of affected symbols
            
        Returns:
            Market event for the broker simulator
        """
        from trading_gateway_service.simulation.forex_broker_simulator import MarketEvent, MarketEventType
        event_type_mapping = {NewsCategory.ECONOMIC_DATA: MarketEventType.
            ECONOMIC_RELEASE, NewsCategory.CENTRAL_BANK: MarketEventType.
            CENTRAL_BANK_DECISION, NewsCategory.GEOPOLITICAL:
            MarketEventType.GEOPOLITICAL_EVENT, NewsCategory.
            NATURAL_DISASTER: MarketEventType.GEOPOLITICAL_EVENT,
            NewsCategory.MARKET_SENTIMENT: MarketEventType.
            TECHNICAL_BREAKOUT, NewsCategory.REGULATORY: MarketEventType.
            CENTRAL_BANK_DECISION, NewsCategory.CORPORATE: MarketEventType.
            ECONOMIC_RELEASE, NewsCategory.COMMODITY: MarketEventType.
            GEOPOLITICAL_EVENT}
        event_type = event_type_mapping.get(news_event.category,
            MarketEventType.ECONOMIC_RELEASE)
        impact_magnitude_mapping = {NewsImpactLevel.LOW: 0.2,
            NewsImpactLevel.MEDIUM: 0.5, NewsImpactLevel.HIGH: 0.8,
            NewsImpactLevel.CRITICAL: 1.0}
        impact_magnitude = impact_magnitude_mapping.get(news_event.
            impact_level, 0.5)
        market_event = MarketEvent(event_type=event_type, impact_magnitude=
            impact_magnitude, duration_minutes=news_event.duration_minutes,
            description=news_event.title, affected_symbols=affected_symbols,
            volatility_factor=news_event.volatility_impact_factor,
            price_impact_factor=news_event.price_impact_factor)
        return market_event

    def _get_affected_symbols(self, currencies: List[str]) ->List[str]:
        """
        Get trading symbols affected by currencies.
        
        Args:
            currencies: List of currency codes
            
        Returns:
            List of affected symbol strings
        """
        all_symbols = list(self.broker_simulator.prices.keys())
        affected_symbols = []
        for symbol in all_symbols:
            parts = symbol.split('/')
            if len(parts) != 2:
                continue
            base, quote = parts
            if base in currencies or quote in currencies:
                affected_symbols.append(symbol)
        return affected_symbols

    def _generate_random_event(self, current_time: datetime) ->NewsEvent:
        """
        Generate a random news event.
        
        Args:
            current_time: Current time
            
        Returns:
            Generated news event
        """
        event_id = str(uuid.uuid4())
        category_weights = {NewsCategory.ECONOMIC_DATA: 0.4, NewsCategory.
            CENTRAL_BANK: 0.2, NewsCategory.GEOPOLITICAL: 0.15,
            NewsCategory.CORPORATE: 0.05, NewsCategory.MARKET_SENTIMENT: 
            0.1, NewsCategory.NATURAL_DISASTER: 0.03, NewsCategory.
            REGULATORY: 0.05, NewsCategory.COMMODITY: 0.02}
        categories = list(category_weights.keys())
        weights = list(category_weights.values())
        category = random.choices(categories, weights=weights, k=1)[0]
        impact_weights = {NewsImpactLevel.LOW: 0.5, NewsImpactLevel.MEDIUM:
            0.3, NewsImpactLevel.HIGH: 0.15, NewsImpactLevel.CRITICAL: 0.05}
        impacts = list(impact_weights.keys())
        impact_probs = list(impact_weights.values())
        impact_level = random.choices(impacts, weights=impact_probs, k=1)[0]
        titles = self._get_random_titles(category)
        title = random.choice(titles)
        descriptions = self._get_random_descriptions(category)
        description = random.choice(descriptions)
        time_offset_minutes = random.randint(5, 60)
        release_time = current_time + timedelta(minutes=time_offset_minutes)
        all_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF',
            'NZD']
        num_affected = random.randint(1, 3)
        affected_currencies = random.sample(all_currencies, num_affected)
        sentiment_impact = {}
        for currency in affected_currencies:
            sentiment_value = random.uniform(-1.0, 1.0)
            sentiment_level = self.sentiment_tracker.value_to_sentiment(
                sentiment_value)
            sentiment_impact[currency] = sentiment_level
        impact_mapping = {NewsImpactLevel.LOW: (0.0001, 0.0003, 1.1, 0.9, 
            30), NewsImpactLevel.MEDIUM: (0.0003, 0.0008, 1.3, 0.8, 60),
            NewsImpactLevel.HIGH: (0.0008, 0.0015, 1.8, 0.7, 120),
            NewsImpactLevel.CRITICAL: (0.0015, 0.003, 2.5, 0.5, 180)}
        min_price, max_price, vol_factor, liq_factor, duration = (
            impact_mapping[impact_level])
        price_impact_factor = random.uniform(min_price, max_price)
        if random.random() < 0.5:
            price_impact_factor *= -1
        technical_impacts = list(TechnicalImpact)
        technical_impact = random.choice(technical_impacts)
        event = NewsEvent(event_id=event_id, category=category,
            impact_level=impact_level, title=title, description=description,
            release_time=release_time, affected_currencies=
            affected_currencies, expected_value=None, actual_value=None,
            previous_value=None, sentiment_impact=sentiment_impact,
            technical_impact=technical_impact, price_impact_factor=
            price_impact_factor, volatility_impact_factor=vol_factor,
            liquidity_impact_factor=liq_factor, duration_minutes=duration)
        self.news_calendar.add_event(event)
        logger.info(f'Generated random news event: {title} at {release_time}')
        return event

    def _get_random_titles(self, category: NewsCategory) ->List[str]:
        """
        Get random news titles for a category.
        
        Args:
            category: News category
            
        Returns:
            List of possible titles
        """
        title_templates = {NewsCategory.ECONOMIC_DATA: [
            '{country} {indicator} {direction} to {value}%',
            '{country} {indicator} {beats_misses} expectations',
            '{country} {indicator} shows {adj} growth',
            '{country} releases {adj} {indicator} data',
            'Unexpected {direction} in {country} {indicator}'],
            NewsCategory.CENTRAL_BANK: [
            '{central_bank} {action} interest rates by {value} basis points',
            '{central_bank} maintains policy rate at {value}%',
            '{central_bank} signals {adj} monetary policy stance',
            '{central_bank} {action} QE program',
            '{central_bank} chair comments on {economic_condition}'],
            NewsCategory.GEOPOLITICAL: [
            '{country} election results show {adj} outcome',
            'Trade tensions between {country} and {country2} {escalate_ease}',
            '{country} announces new {policy_type} policy',
            'Political uncertainty in {country} affects markets',
            '{country} {action} new sanctions against {country2}'],
            NewsCategory.CORPORATE: [
            'Major {industry} firms report {adj} earnings',
            '{company} announces {adj} quarterly results',
            '{industry} sector shows signs of {economic_condition}',
            '{company} {action} merger with {company2}',
            'Corporate bond yields {direction} as {economic_condition} concerns grow'
            ], NewsCategory.MARKET_SENTIMENT: [
            'Market sentiment turns {sentiment} as {economic_condition}',
            'Risk {appetite_aversion} dominates market sentiment',
            'Investors turn {sentiment} on {currency} outlook',
            'Global markets show {adj} sentiment shift',
            '{sentiment} sentiment prevails despite {economic_condition}'],
            NewsCategory.NATURAL_DISASTER: [
            '{disaster} in {country} disrupts financial markets',
            '{disaster} impacts {country} economic outlook',
            '{country} faces economic consequences from {disaster}',
            'Markets react to {disaster} in {country}',
            '{disaster} in {country} threatens supply chains'],
            NewsCategory.REGULATORY: [
            'New {regulatory_body} regulations impact {industry} sector',
            '{country} introduces new {policy_type} policies',
            'Regulatory changes affect {industry} outlook',
            '{regulatory_body} announces new {policy_type} framework',
            '{country} reforms {policy_type} regulations'], NewsCategory.
            COMMODITY: ['Oil prices {direction} amid {economic_condition}',
            'Gold {direction} as {sentiment} sentiment prevails',
            'Commodity markets react to {economic_condition}',
            '{commodity} prices hit {timeframe} {high_low} on {economic_condition}'
            , '{country} {action} {commodity} production levels']}
        templates = title_templates.get(category, [
            'Economic news impacts markets'])
        filled_templates = []
        for template in templates:
            filled = template.format(country=random.choice(['US', 'EU',
                'UK', 'Japan', 'China', 'Australia', 'Canada',
                'Switzerland']), country2=random.choice(['US', 'EU', 'UK',
                'Japan', 'China', 'Russia', 'Brazil', 'India']), indicator=
                random.choice(['GDP', 'Inflation', 'CPI', 'PPI',
                'Unemployment', 'Retail Sales', 'PMI',
                'Industrial Production']), direction=random.choice(['rises',
                'falls', 'jumps', 'plummets', 'increases', 'decreases']),
                value=random.randint(1, 10) / 10.0, beats_misses=random.
                choice(['beats', 'misses', 'matches']), adj=random.choice([
                'strong', 'weak', 'moderate', 'robust', 'sluggish',
                'unexpected', 'anticipated']), central_bank=random.choice([
                'Fed', 'ECB', 'BoE', 'BoJ', 'RBA', 'BoC', 'SNB', 'PBOC']),
                action=random.choice(['raises', 'cuts', 'maintains',
                'increases', 'decreases', 'expands', 'reduces']),
                escalate_ease=random.choice(['escalate', 'ease',
                'intensify', 'improve']), policy_type=random.choice([
                'fiscal', 'monetary', 'trade', 'tax', 'economic']),
                industry=random.choice(['banking', 'tech', 'energy',
                'retail', 'automotive', 'healthcare', 'manufacturing']),
                company=random.choice(['Major bank', 'Tech giant',
                'Energy firm', 'Retail leader', 'Auto manufacturer']),
                company2=random.choice(['competitor', 'industry leader',
                'global player', 'market entrant']), economic_condition=
                random.choice(['inflation concerns', 'growth outlook',
                'recession fears', 'recovery hopes',
                'stability expectations']), sentiment=random.choice([
                'bullish', 'bearish', 'optimistic', 'pessimistic',
                'cautious']), appetite_aversion=random.choice(['appetite',
                'aversion']), disaster=random.choice(['Earthquake',
                'Hurricane', 'Flood', 'Drought', 'Wildfire', 'Pandemic']),
                regulatory_body=random.choice(['Central Bank', 'SEC',
                'CFTC', 'FSA', 'Government', 'Treasury']), commodity=random
                .choice(['Oil', 'Gold', 'Natural Gas', 'Copper',
                'Agricultural products', 'Metals']), timeframe=random.
                choice(['yearly', 'monthly', 'weekly', 'daily']), high_low=
                random.choice(['high', 'low']), currency=random.choice([
                'USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF']))
            filled_templates.append(filled)
        return filled_templates

    def _get_random_descriptions(self, category: NewsCategory) ->List[str]:
        """
        Get random news descriptions for a category.
        
        Args:
            category: News category
            
        Returns:
            List of possible descriptions
        """
        descriptions = [
            'The recent economic data has significant implications for currency markets.'
            ,
            'Analysts are closely watching the impact of this development on financial markets.'
            ,
            'Market participants are adjusting positions in response to this announcement.'
            ,
            'This development could influence central bank policy decisions in coming months.'
            ,
            'Traders are responding to the unexpected nature of this economic indicator.'
            ,
            'The announcement has created volatility across major currency pairs.'
            ,
            'Economists are revising their forecasts based on this new information.'
            ,
            'This event may lead to significant repricing in the forex market.'
            ,
            'Market reactions suggest this news was not fully priced in by traders.'
            , 'Long-term implications of this development remain uncertain.']
        return descriptions

    def generate_economic_calendar(self, start_date: datetime, end_date:
        datetime, num_events: int=10) ->None:
        """
        Generate a realistic economic calendar for a time period.
        
        Args:
            start_date: Start date for calendar
            end_date: End date for calendar
            num_events: Number of events to generate
        """
        self.news_calendar = NewsCalendar()
        self._generate_regular_economic_events(start_date, end_date)
        for _ in range(num_events):
            time_range = (end_date - start_date).total_seconds()
            random_offset = random.uniform(0, time_range)
            event_time = start_date + timedelta(seconds=random_offset)
            self._generate_random_event(event_time)

    def _generate_regular_economic_events(self, start_date: datetime,
        end_date: datetime) ->None:
        """
        Generate regular economic events (e.g., monthly jobs reports, central bank meetings).
        
        Args:
            start_date: Start date
            end_date: End date
        """
        fed_meetings = [(2025, 1, 29), (2025, 3, 19), (2025, 4, 30), (2025,
            6, 11), (2025, 7, 30), (2025, 9, 17), (2025, 11, 5), (2025, 12, 17)
            ]
        for year, month, day in fed_meetings:
            meeting_date = datetime(year, month, day, 14, 0)
            if start_date <= meeting_date <= end_date:
                event = NewsEvent(event_id=str(uuid.uuid4()), category=
                    NewsCategory.CENTRAL_BANK, impact_level=NewsImpactLevel
                    .HIGH, title='Fed Interest Rate Decision', description=
                    'The Federal Reserve announces its monetary policy decision.'
                    , release_time=meeting_date, affected_currencies=['USD'
                    ], expected_value=None, actual_value=None,
                    previous_value=None, sentiment_impact={'USD':
                    SentimentLevel.NEUTRAL}, technical_impact=
                    TechnicalImpact.VOLATILITY_SPIKE, price_impact_factor=
                    0.0, volatility_impact_factor=2.0,
                    liquidity_impact_factor=0.8, duration_minutes=120)
                self.news_calendar.add_event(event)
        current_month = start_date.replace(day=1)
        while current_month <= end_date:
            first_day = current_month.weekday()
            days_until_friday = (4 - first_day) % 7
            first_friday = current_month + timedelta(days=days_until_friday)
            if first_friday < start_date:
                first_friday = first_friday + timedelta(days=7)
            if first_friday <= end_date:
                nfp_date = first_friday.replace(hour=8, minute=30)
                event = NewsEvent(event_id=str(uuid.uuid4()), category=
                    NewsCategory.ECONOMIC_DATA, impact_level=
                    NewsImpactLevel.HIGH, title='US Non-Farm Payrolls',
                    description=
                    'Monthly employment data showing job creation in the US economy.'
                    , release_time=nfp_date, affected_currencies=['USD'],
                    expected_value=None, actual_value=None, previous_value=
                    None, sentiment_impact={'USD': SentimentLevel.NEUTRAL},
                    technical_impact=TechnicalImpact.VOLATILITY_SPIKE,
                    price_impact_factor=0.0, volatility_impact_factor=2.2,
                    liquidity_impact_factor=0.7, duration_minutes=90)
                self.news_calendar.add_event(event)
            if current_month.month == 12:
                current_month = current_month.replace(year=current_month.
                    year + 1, month=1)
            else:
                current_month = current_month.replace(month=current_month.
                    month + 1)

    def add_scheduled_event(self, category: NewsCategory, impact_level:
        NewsImpactLevel, title: str, description: str, release_time:
        datetime, affected_currencies: List[str], technical_impact:
        TechnicalImpact=TechnicalImpact.NONE, price_impact_factor: float=
        0.0, volatility_impact_factor: float=1.0, duration_minutes: int=60
        ) ->NewsEvent:
        """
        Add a scheduled event to the calendar.
        
        Args:
            category: News category
            impact_level: Impact level
            title: Event title
            description: Event description
            release_time: Release time
            affected_currencies: List of affected currencies
            technical_impact: Technical impact type
            price_impact_factor: Price impact factor
            volatility_impact_factor: Volatility impact factor
            duration_minutes: Duration in minutes
            
        Returns:
            Created news event
        """
        sentiment_impact = {}
        for currency in affected_currencies:
            sentiment_impact[currency] = SentimentLevel.NEUTRAL
        event = NewsEvent(event_id=str(uuid.uuid4()), category=category,
            impact_level=impact_level, title=title, description=description,
            release_time=release_time, affected_currencies=
            affected_currencies, expected_value=None, actual_value=None,
            previous_value=None, sentiment_impact=sentiment_impact,
            technical_impact=technical_impact, price_impact_factor=
            price_impact_factor, volatility_impact_factor=
            volatility_impact_factor, liquidity_impact_factor=1.0 - (
            volatility_impact_factor - 1.0) * 0.5, duration_minutes=
            duration_minutes)
        self.news_calendar.add_event(event)
        return event

    @with_broker_api_resilience('get_sentiment_data')
    def get_sentiment_data(self, symbols: List[str], lookback: int=10) ->Dict[
        str, Dict[str, Any]]:
        """
        Get sentiment data for trading symbols.
        
        Args:
            symbols: List of trading symbols
            lookback: Number of historical points
            
        Returns:
            Dictionary of sentiment data by symbol
        """
        sentiment_data = {}
        for symbol in symbols:
            parts = symbol.split('/')
            if len(parts) != 2:
                continue
            base, quote = parts
            current_sentiment = self.sentiment_tracker.get_pair_sentiment(base,
                quote)
            current_value = self.sentiment_tracker.sentiment_to_value(
                current_sentiment)
            base_history = self.sentiment_tracker.get_historical_sentiment(base
                , lookback)
            quote_history = self.sentiment_tracker.get_historical_sentiment(
                quote, lookback)
            pair_sentiment_history = []
            for i in range(min(len(base_history), len(quote_history))):
                base_value = self.sentiment_tracker.sentiment_to_value(
                    base_history[i])
                quote_value = self.sentiment_tracker.sentiment_to_value(
                    quote_history[i])
                pair_value = base_value - quote_value
                pair_value = max(-1.0, min(1.0, pair_value))
                pair_sentiment_history.append(self.sentiment_tracker.
                    value_to_sentiment(pair_value))
            sentiment_data[symbol] = {'current_sentiment':
                current_sentiment.value, 'current_value': current_value,
                'sentiment_history': [s.value for s in
                pair_sentiment_history], 'base_currency_sentiment': self.
                sentiment_tracker.get_current_sentiment(base).value,
                'quote_currency_sentiment': self.sentiment_tracker.
                get_current_sentiment(quote).value}
        return sentiment_data

    @with_broker_api_resilience('get_upcoming_events')
    def get_upcoming_events(self, hours_ahead: int=24, impact_filter:
        Optional[List[NewsImpactLevel]]=None) ->List[Dict[str, Any]]:
        """
        Get upcoming scheduled events.
        
        Args:
            hours_ahead: How many hours to look ahead
            impact_filter: Optional filter for impact levels
            
        Returns:
            List of upcoming events as dictionaries
        """
        start_time = self.current_time
        end_time = start_time + timedelta(hours=hours_ahead)
        events = self.news_calendar.get_events_in_window(start_time, end_time)
        if impact_filter:
            events = [e for e in events if e.impact_level in impact_filter]
        event_dicts = []
        for event in events:
            event_dict = {'id': event.event_id, 'title': event.title,
                'category': event.category.value, 'impact_level': event.
                impact_level.value, 'release_time': event.release_time.
                isoformat(), 'affected_currencies': event.
                affected_currencies, 'time_until_minutes': int((event.
                release_time - self.current_time).total_seconds() / 60)}
            event_dicts.append(event_dict)
        event_dicts.sort(key=lambda x: x['release_time'])
        return event_dicts
