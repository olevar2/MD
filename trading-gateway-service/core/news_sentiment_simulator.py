"""
News and Sentiment Impact Simulator for Forex Trading Platform.

This module simulates the impact of news events and sentiment on forex markets,
providing realistic price movements for economic releases, central bank decisions,
and sentiment shifts.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import uuid
from enum import Enum
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from utils.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class NewsImpactLevel(Enum):
    """Enum representing different levels of news impact."""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'


class NewsEventType(Enum):
    """Enum representing different types of news events."""
    ECONOMIC_DATA = 'economic_data'
    CENTRAL_BANK = 'central_bank'
    GEOPOLITICAL = 'geopolitical'
    EARNINGS = 'earnings'
    NATURAL_DISASTER = 'natural_disaster'
    MARKET_SENTIMENT = 'market_sentiment'


class SentimentLevel(Enum):
    """Enum representing different market sentiment levels."""
    VERY_BEARISH = 'very_bearish'
    BEARISH = 'bearish'
    SLIGHTLY_BEARISH = 'slightly_bearish'
    NEUTRAL = 'neutral'
    SLIGHTLY_BULLISH = 'slightly_bullish'
    BULLISH = 'bullish'
    VERY_BULLISH = 'very_bullish'


class NewsEvent:
    """Class representing a news event with market impact."""

    def __init__(self, event_id: str, event_type: NewsEventType,
        impact_level: NewsImpactLevel, timestamp: datetime,
        currencies_affected: List[str], title: str, description: str='',
        expected_value: Optional[float]=None, actual_value: Optional[float]
        =None, previous_value: Optional[float]=None, sentiment_impact:
        Optional[SentimentLevel]=None, volatility_impact: float=0.0,
        price_impact: float=0.0, duration_minutes: int=60):
        """
        Initialize a news event.
        
        Args:
            event_id: Unique identifier for the event
            event_type: Type of news event
            impact_level: Level of market impact
            timestamp: When the event occurs
            currencies_affected: List of currency pairs affected
            title: Event title
            description: Optional detailed description
            expected_value: Expected value for economic data
            actual_value: Actual released value
            previous_value: Previous value for comparison
            sentiment_impact: Impact on market sentiment
            volatility_impact: Impact on market volatility (multiplier)
            price_impact: Direct price impact (percentage)
            duration_minutes: How long the event impact lasts
        """
        self.event_id = event_id or str(uuid.uuid4())
        self.event_type = event_type
        self.impact_level = impact_level
        self.timestamp = timestamp
        self.currencies_affected = currencies_affected
        self.title = title
        self.description = description
        self.expected_value = expected_value
        self.actual_value = actual_value
        self.previous_value = previous_value
        self.sentiment_impact = sentiment_impact
        self.volatility_impact = volatility_impact
        self.price_impact = price_impact
        self.duration_minutes = duration_minutes

    def has_surprise(self) ->bool:
        """Check if the event has a surprise versus expectations."""
        if self.expected_value is None or self.actual_value is None:
            return False
        surprise_pct = abs(self.actual_value - self.expected_value) / abs(self
            .expected_value) if self.expected_value != 0 else 0
        return surprise_pct > 0.1

    def is_positive(self) ->bool:
        """Check if the event is positive for the currency."""
        if self.sentiment_impact:
            return self.sentiment_impact in [SentimentLevel.
                SLIGHTLY_BULLISH, SentimentLevel.BULLISH, SentimentLevel.
                VERY_BULLISH]
        if self.expected_value is not None and self.actual_value is not None:
            return self.actual_value > self.expected_value
        return self.price_impact > 0

    @with_broker_api_resilience('get_impact_factor')
    def get_impact_factor(self) ->float:
        """Get a numeric impact factor based on the impact level."""
        impact_factors = {NewsImpactLevel.LOW: 0.2, NewsImpactLevel.MEDIUM:
            0.5, NewsImpactLevel.HIGH: 1.0, NewsImpactLevel.CRITICAL: 2.0}
        return impact_factors.get(self.impact_level, 0.5)

    def to_dict(self) ->Dict:
        """Convert to dictionary for serialization."""
        return {'event_id': self.event_id, 'event_type': self.event_type.
            value, 'impact_level': self.impact_level.value, 'timestamp':
            self.timestamp.isoformat(), 'currencies_affected': self.
            currencies_affected, 'title': self.title, 'description': self.
            description, 'expected_value': self.expected_value,
            'actual_value': self.actual_value, 'previous_value': self.
            previous_value, 'sentiment_impact': self.sentiment_impact.value if
            self.sentiment_impact else None, 'volatility_impact': self.
            volatility_impact, 'price_impact': self.price_impact,
            'duration_minutes': self.duration_minutes}

    @classmethod
    def from_dict(cls, data: Dict) ->'NewsEvent':
        """Create from dictionary."""
        event_type = NewsEventType(data['event_type'])
        impact_level = NewsImpactLevel(data['impact_level'])
        sentiment_impact = SentimentLevel(data['sentiment_impact']
            ) if data.get('sentiment_impact') else None
        return cls(event_id=data['event_id'], event_type=event_type,
            impact_level=impact_level, timestamp=datetime.fromisoformat(
            data['timestamp']), currencies_affected=data[
            'currencies_affected'], title=data['title'], description=data.
            get('description', ''), expected_value=data.get(
            'expected_value'), actual_value=data.get('actual_value'),
            previous_value=data.get('previous_value'), sentiment_impact=
            sentiment_impact, volatility_impact=data.get(
            'volatility_impact', 0.0), price_impact=data.get('price_impact',
            0.0), duration_minutes=data.get('duration_minutes', 60))


class NewsAndSentimentSimulator:
    """
    Simulator for news events and sentiment impact on forex markets.
    
    This class models realistic market reactions to economic releases, central bank
    decisions, and shifts in market sentiment.
    """

    def __init__(self, seed: Optional[int]=None):
        """
        Initialize the news and sentiment simulator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self.current_time = datetime.now()
        self.news_calendar = []
        self.active_events = []
        self.sentiment_by_currency = {}

    def set_current_time(self, time: datetime) ->None:
        """
        Set the current simulation time.
        
        Args:
            time: Current datetime for the simulation
        """
        self.current_time = time
        self._update_active_events()

    def advance_time(self, seconds: int) ->None:
        """
        Advance the simulation time.
        
        Args:
            seconds: Number of seconds to advance
        """
        self.current_time += timedelta(seconds=seconds)
        self._update_active_events()

    def _update_active_events(self) ->None:
        """Update the list of active events based on the current time."""
        self.active_events = [event for event in self.active_events if 
            event.timestamp + timedelta(minutes=event.duration_minutes) >
            self.current_time]
        new_events = []
        for event in self.news_calendar:
            if (event.timestamp <= self.current_time and event.timestamp +
                timedelta(minutes=event.duration_minutes) > self.
                current_time and event not in self.active_events):
                new_events.append(event)
        if new_events:
            self.active_events.extend(new_events)
            logger.info(f'Added {len(new_events)} new active events')

    def add_news_event(self, event: NewsEvent) ->None:
        """
        Add a news event to the calendar.
        
        Args:
            event: The news event to add
        """
        self.news_calendar.append(event)
        logger.info(f'Added news event: {event.title} at {event.timestamp}')
        self._update_active_events()

    def generate_random_economic_calendar(self, start_date: datetime,
        end_date: datetime, currency_pairs: List[str], num_events: int=20
        ) ->List[NewsEvent]:
        """
        Generate a random economic calendar for simulation.
        
        Args:
            start_date: Start date for the calendar
            end_date: End date for the calendar
            currency_pairs: Currency pairs to generate events for
            num_events: Number of events to generate
            
        Returns:
            List of generated NewsEvent objects
        """
        events = []
        event_types = list(NewsEventType)
        impact_levels = list(NewsImpactLevel)
        sentiment_levels = list(SentimentLevel)
        type_weights = [0.5, 0.25, 0.1, 0.05, 0.05, 0.05]
        impact_weights = [0.5, 0.3, 0.15, 0.05]
        event_titles = {NewsEventType.ECONOMIC_DATA: ['Non-Farm Payrolls',
            'GDP Growth', 'CPI Inflation', 'Retail Sales',
            'PMI Manufacturing', 'PMI Services', 'Unemployment Rate',
            'Trade Balance', 'Industrial Production', 'Building Permits',
            'Consumer Confidence'], NewsEventType.CENTRAL_BANK: [
            'Interest Rate Decision', 'Monetary Policy Statement',
            'Central Bank Minutes', 'Governor Speech',
            'Quantitative Easing Adjustment', 'Forward Guidance Update'],
            NewsEventType.GEOPOLITICAL: ['Election Results',
            'Trade Agreement', 'Brexit Developments', 'Tariff Announcement',
            'Diplomatic Tensions', 'Military Conflict'], NewsEventType.
            EARNINGS: ['Major Bank Earnings', 'Tech Sector Results',
            'Earnings Season Summary', 'Financial Sector Outlook',
            'Corporate Guidance Update'], NewsEventType.NATURAL_DISASTER: [
            'Hurricane Impact', 'Earthquake Damage Assessment',
            'Flooding Economic Impact', 'Wildfire Damage Report'],
            NewsEventType.MARKET_SENTIMENT: ['Risk Appetite Shift',
            'Safe Haven Demand', 'Dollar Sentiment',
            'Market Positioning Data', 'Speculative Positioning Change']}
        time_range = (end_date - start_date).total_seconds()
        timestamps = [(start_date + timedelta(seconds=self.rng.randint(0,
            time_range))) for _ in range(num_events)]
        timestamps.sort()
        for i, timestamp in enumerate(timestamps):
            event_type = self.rng.choice(event_types, p=type_weights)
            impact_level = self.rng.choice(impact_levels, p=impact_weights)
            num_currencies = self.rng.randint(1, min(4, len(currency_pairs) +
                1))
            affected_currencies = self.rng.choice(currency_pairs, size=
                num_currencies, replace=False).tolist()
            titles = event_titles.get(event_type, ['News Event'])
            title = self.rng.choice(titles)
            if event_type == NewsEventType.ECONOMIC_DATA:
                main_currency = affected_currencies[0].split('/')[0]
                title = f'{main_currency} {title}'
            expected_value = None
            actual_value = None
            previous_value = None
            if event_type == NewsEventType.ECONOMIC_DATA:
                base_values = {'Non-Farm Payrolls': 200000, 'GDP Growth': 
                    2.5, 'CPI Inflation': 2.0, 'Retail Sales': 0.4,
                    'PMI Manufacturing': 52, 'PMI Services': 54,
                    'Unemployment Rate': 4.0, 'Trade Balance': -50,
                    'Industrial Production': 0.3, 'Building Permits': 1.8,
                    'Consumer Confidence': 100}
                base = base_values.get(title.split(' ', 1)[1] if ' ' in
                    title else title, 50)
                variation = base * 0.1
                previous_value = base + self.rng.normal(0, variation / 2)
                expected_value = previous_value + self.rng.normal(0, 
                    variation / 3)
                if self.rng.random() < 0.3:
                    surprise_factor = 3 if self.rng.random() < 0.5 else -3
                    actual_value = expected_value + self.rng.normal(
                        surprise_factor * variation / 3, variation / 4)
                else:
                    actual_value = expected_value + self.rng.normal(0, 
                        variation / 4)
            sentiment_probs = [0.05, 0.1, 0.15, 0.4, 0.15, 0.1, 0.05]
            sentiment_impact = self.rng.choice(sentiment_levels, p=
                sentiment_probs)
            base_impact = 0.0
            if impact_level == NewsImpactLevel.LOW:
                base_impact = self.rng.normal(0, 0.0005)
            elif impact_level == NewsImpactLevel.MEDIUM:
                base_impact = self.rng.normal(0, 0.0015)
            elif impact_level == NewsImpactLevel.HIGH:
                base_impact = self.rng.normal(0, 0.003)
            else:
                base_impact = self.rng.normal(0, 0.01)
            sentiment_factor = 0.0
            if sentiment_impact == SentimentLevel.VERY_BEARISH:
                sentiment_factor = -2.0
            elif sentiment_impact == SentimentLevel.BEARISH:
                sentiment_factor = -1.0
            elif sentiment_impact == SentimentLevel.SLIGHTLY_BEARISH:
                sentiment_factor = -0.5
            elif sentiment_impact == SentimentLevel.SLIGHTLY_BULLISH:
                sentiment_factor = 0.5
            elif sentiment_impact == SentimentLevel.BULLISH:
                sentiment_factor = 1.0
            elif sentiment_impact == SentimentLevel.VERY_BULLISH:
                sentiment_factor = 2.0
            price_impact = base_impact * (1 + 0.5 * sentiment_factor)
            volatility_impact = 1.0
            if impact_level == NewsImpactLevel.LOW:
                volatility_impact = 1.2
            elif impact_level == NewsImpactLevel.MEDIUM:
                volatility_impact = 1.5
            elif impact_level == NewsImpactLevel.HIGH:
                volatility_impact = 2.0
            else:
                volatility_impact = 3.0
            duration_map = {NewsImpactLevel.LOW: self.rng.randint(15, 60),
                NewsImpactLevel.MEDIUM: self.rng.randint(30, 180),
                NewsImpactLevel.HIGH: self.rng.randint(60, 360),
                NewsImpactLevel.CRITICAL: self.rng.randint(120, 1440)}
            duration_minutes = duration_map[impact_level]
            event = NewsEvent(event_id=f'event_{i}', event_type=event_type,
                impact_level=impact_level, timestamp=timestamp,
                currencies_affected=affected_currencies, title=title,
                description=f'Simulated {event_type.value} event',
                expected_value=expected_value, actual_value=actual_value,
                previous_value=previous_value, sentiment_impact=
                sentiment_impact, volatility_impact=volatility_impact,
                price_impact=price_impact, duration_minutes=duration_minutes)
            events.append(event)
        self.news_calendar = events
        return events

    @with_broker_api_resilience('get_active_events')
    def get_active_events(self, currency_pair: Optional[str]=None) ->List[
        NewsEvent]:
        """
        Get a list of currently active news events.
        
        Args:
            currency_pair: Optional filter for specific currency pair
            
        Returns:
            List of active events affecting the currency pair
        """
        if currency_pair:
            return [event for event in self.active_events if currency_pair in
                event.currencies_affected]
        return self.active_events

    @with_market_data_resilience('calculate_price_impact')
    def calculate_price_impact(self, currency_pair: str, base_price: float,
        current_volatility: float=0.0001) ->Dict[str, float]:
        """
        Calculate price impact from active news events.
        
        Args:
            currency_pair: Currency pair to calculate for
            base_price: Current base price
            current_volatility: Current market volatility
            
        Returns:
            Dictionary with price impact metrics
        """
        active_events = self.get_active_events(currency_pair)
        if not active_events:
            return {'price_change_pct': 0.0, 'volatility_multiplier': 1.0,
                'spread_multiplier': 1.0}
        price_impact = 0.0
        volatility_impact = 1.0
        spread_impact = 1.0
        for event in active_events:
            elapsed_minutes = (self.current_time - event.timestamp
                ).total_seconds() / 60
            time_factor = max(0, 1 - elapsed_minutes / event.duration_minutes)
            price_impact += event.price_impact * time_factor
            vol_factor = 1.0 + (event.volatility_impact - 1.0) * time_factor
            volatility_impact *= vol_factor
            spread_factor = 1.0 + (vol_factor - 1.0) * 0.7
            spread_impact *= spread_factor
        volatility_impact = min(volatility_impact, 5.0)
        spread_impact = min(spread_impact, 5.0)
        return {'price_change_pct': price_impact, 'volatility_multiplier':
            volatility_impact, 'spread_multiplier': spread_impact}

    @with_broker_api_resilience('calculate_gap_probability')
    def calculate_gap_probability(self, currency_pair: str) ->float:
        """
        Calculate probability of price gaps due to news events.
        
        Args:
            currency_pair: Currency pair to check
            
        Returns:
            Probability of a price gap (0.0-1.0)
        """
        active_events = self.get_active_events(currency_pair)
        if not active_events:
            return 0.01
        gap_prob = 0.01
        for event in active_events:
            if currency_pair in event.currencies_affected:
                impact_factor = event.get_impact_factor()
                elapsed_minutes = (self.current_time - event.timestamp
                    ).total_seconds() / 60
                time_factor = max(0, 1 - elapsed_minutes / event.
                    duration_minutes)
                additional_prob = 0.05 * impact_factor * time_factor
                gap_prob = (gap_prob + additional_prob - gap_prob *
                    additional_prob)
        return min(gap_prob, 0.5)

    def generate_gap_size(self, currency_pair: str, base_price: float) ->float:
        """
        Generate a price gap size if a gap occurs.
        
        Args:
            currency_pair: Currency pair to check
            base_price: Base price for calculating gap size
            
        Returns:
            Price gap size (can be positive or negative)
        """
        active_events = self.get_active_events(currency_pair)
        base_gap = base_price * 0.0005
        if not active_events:
            direction = 1 if self.rng.random() > 0.5 else -1
            return direction * base_gap * self.rng.lognormal(0, 0.5)
        cumulative_impact = 0.0
        max_impact_factor = 0.0
        for event in active_events:
            if currency_pair in event.currencies_affected:
                impact_factor = event.get_impact_factor()
                max_impact_factor = max(max_impact_factor, impact_factor)
                direction = 1 if event.is_positive() else -1
                elapsed_minutes = (self.current_time - event.timestamp
                    ).total_seconds() / 60
                time_factor = max(0, 1 - elapsed_minutes / event.
                    duration_minutes)
                cumulative_impact += direction * impact_factor * time_factor
        gap_multiplier = max_impact_factor * (1 + abs(cumulative_impact))
        gap_size = base_gap * gap_multiplier * self.rng.lognormal(0, 0.3)
        direction = 1 if cumulative_impact > 0 else -1
        return direction * gap_size

    @with_broker_api_resilience('calculate_slippage_impact')
    def calculate_slippage_impact(self, currency_pair: str, order_size:
        float, base_slippage: float) ->float:
        """
        Calculate slippage impact during news events.
        
        Args:
            currency_pair: Currency pair for the order
            order_size: Size of the order in lots
            base_slippage: Base slippage without news impact
            
        Returns:
            Modified slippage value
        """
        active_events = self.get_active_events(currency_pair)
        if not active_events:
            return base_slippage
        slippage_multiplier = 1.0
        for event in active_events:
            if currency_pair in event.currencies_affected:
                impact_factor = event.get_impact_factor()
                elapsed_minutes = (self.current_time - event.timestamp
                    ).total_seconds() / 60
                time_factor = max(0, 1 - elapsed_minutes / event.
                    duration_minutes)
                additional_factor = 1.0 + impact_factor * time_factor
                slippage_multiplier *= additional_factor
        size_factor = 1.0 + 0.5 * (order_size - 1)
        return base_slippage * slippage_multiplier * size_factor

    def set_sentiment(self, currency: str, sentiment: SentimentLevel) ->None:
        """
        Set the overall sentiment for a currency.
        
        Args:
            currency: The currency code (e.g., 'USD', 'EUR')
            sentiment: The sentiment level
        """
        self.sentiment_by_currency[currency] = sentiment
        logger.info(f'Set sentiment for {currency} to {sentiment.value}')

    @with_broker_api_resilience('get_sentiment')
    def get_sentiment(self, currency: str) ->SentimentLevel:
        """
        Get the overall sentiment for a currency.
        
        Args:
            currency: The currency code (e.g., 'USD', 'EUR')
            
        Returns:
            Sentiment level for the currency
        """
        return self.sentiment_by_currency.get(currency, SentimentLevel.NEUTRAL)

    @with_broker_api_resilience('get_pair_sentiment_impact')
    def get_pair_sentiment_impact(self, currency_pair: str) ->float:
        """
        Get the sentiment impact value for a currency pair.
        
        Args:
            currency_pair: Currency pair (e.g., 'EUR/USD')
            
        Returns:
            Sentiment impact as a float (-1.0 to 1.0)
        """
        base_currency, quote_currency = currency_pair.split('/')
        base_sentiment = self.get_sentiment(base_currency)
        quote_sentiment = self.get_sentiment(quote_currency)
        sentiment_values = {SentimentLevel.VERY_BEARISH: -3.0,
            SentimentLevel.BEARISH: -2.0, SentimentLevel.SLIGHTLY_BEARISH: 
            -1.0, SentimentLevel.NEUTRAL: 0.0, SentimentLevel.
            SLIGHTLY_BULLISH: 1.0, SentimentLevel.BULLISH: 2.0,
            SentimentLevel.VERY_BULLISH: 3.0}
        base_value = sentiment_values.get(base_sentiment, 0.0)
        quote_value = sentiment_values.get(quote_sentiment, 0.0)
        pair_sentiment = base_value - quote_value
        normalized_sentiment = max(min(pair_sentiment / 3.0, 1.0), -1.0)
        return normalized_sentiment

    def generate_sentiment_driven_momentum(self, currency_pair: str,
        base_price: float, lookback_window: int=10) ->float:
        """
        Generate price momentum based on sentiment.
        
        Args:
            currency_pair: Currency pair to generate for
            base_price: Current base price
            lookback_window: Number of periods for momentum calculation
            
        Returns:
            Price drift component (percentage)
        """
        sentiment_impact = self.get_pair_sentiment_impact(currency_pair)
        base_drift = 0.0001
        sentiment_drift = base_drift * sentiment_impact
        noise = self.rng.normal(0, 0.5 * abs(sentiment_drift) + 1e-05)
        return sentiment_drift + noise


if __name__ == '__main__':
    simulator = NewsAndSentimentSimulator(seed=42)
    start_date = datetime.now()
    end_date = start_date + timedelta(days=7)
    currency_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD']
    events = simulator.generate_random_economic_calendar(start_date,
        end_date, currency_pairs, num_events=20)
    print(f'Generated {len(events)} news events')
    simulator.set_sentiment('USD', SentimentLevel.SLIGHTLY_BEARISH)
    simulator.set_sentiment('EUR', SentimentLevel.BULLISH)
    for i in range(5):
        if i < len(events):
            simulator.set_current_time(events[i].timestamp)
            print(f'\nEvent: {events[i].title} at {events[i].timestamp}')
            print(
                f'Type: {events[i].event_type.value}, Impact: {events[i].impact_level.value}'
                )
            for pair in currency_pairs:
                impact = simulator.calculate_price_impact(pair, 1.0)
                if abs(impact['price_change_pct']) > 0.0001 or impact[
                    'volatility_multiplier'] > 1.01:
                    print(
                        f"  {pair}: Price impact: {impact['price_change_pct'] * 100:.4f}%, Volatility: x{impact['volatility_multiplier']:.2f}, Spread: x{impact['spread_multiplier']:.2f}"
                        )
                    gap_prob = simulator.calculate_gap_probability(pair)
                    if gap_prob > 0.01:
                        print(f'    Gap probability: {gap_prob * 100:.1f}%')
                        gap_size = simulator.generate_gap_size(pair, 1.0)
                        print(f'    Example gap size: {gap_size * 100:.4f}%')
                        slippage = simulator.calculate_slippage_impact(pair,
                            2.0, 0.2)
                        print(
                            f'    Slippage for 2 lot order: {slippage:.1f} pips'
                            )
""""""
